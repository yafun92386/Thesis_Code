from rouge import Rouge
from itertools import takewhile
from tensorboardX import SummaryWriter 
from nltk.translate.bleu_score import sentence_bleu
import time, math, os, torch, numpy, random

import warnings
warnings.simplefilter("ignore", UserWarning)

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

def asMinutes(s):
	h = math.floor(s / 60 / 60)
	s -= h * 60 * 60
	m = math.floor(s / 60)
	s -= m * 60
	return '%dh %dm %ds' % (h, m, s)

def timeSince(since):
	now = time.time()
	s = now - since
	return '%s' % (asMinutes(s))

def wrtLoss(epoch, train_loss, save_path, train_time):
	with open(save_path + "/total_loss.txt", 'a') as f:
		f.write("Epoch %d : %f (%s)\n" % (epoch, train_loss, train_time))

def trainEpochs(save_dir, data_mode, data_loader, vocab, tokenizer, 
				attn_model,encoder, decoder, 
				encoder_optimizer, decoder_optimizer, it_percent, 
				checkpoint_epoch, num_epochs, teacher_forcing_ratio, data_sel):

	save_path = os.path.join(save_dir)
	if not os.path.exists(save_path):
		os.makedirs(save_path)

	writer = SummaryWriter(save_path)

	criterion = torch.nn.NLLLoss(ignore_index=0)

	start_train = time.time()
	# epoch_loss_train = []
	# epoch_loss_eval = []

	for epoch in range(checkpoint_epoch + 1, num_epochs + 1):

		start_epoch = time.time()
		iter_loss = 0
		iter_loss_total = []

		for it, data in enumerate(data_loader):

			if float(it / len(data_loader)) > float(it_percent):
				break

			# Initialize variables
			loss = 0.0
			encoder_optimizer.zero_grad()
			decoder_optimizer.zero_grad()

			# preprocessing
			if data_mode == "jieba":
				loss, _ = jieba_train(data_mode, data, encoder, decoder, attn_model, teacher_forcing_ratio, loss, criterion)
				# sent_tensors, qus_tensors = [d.to(device) for d in data]
			elif data_mode == "bert":
				loss, _ = bert_train(data_mode, data, encoder, decoder, attn_model, teacher_forcing_ratio, loss, criterion)

			# Perform backpropatation
			loss.backward()
			# Adjust model weights
			encoder_optimizer.step()
			decoder_optimizer.step()

			# each iter
			if data_mode == "jieba":
				iter_loss += float(loss.item() / len(data[0]))
			elif data_mode == "bert":
				iter_loss += float(loss.item() / len(data[0][0]))

			# Print progress
			if (it % 100 == 0) & (it != 0):
				iter_loss_avg = iter_loss / 100
				print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(it, it / len(data_loader) * 100, iter_loss_avg))
				iter_loss_total.append(iter_loss_avg)
				iter_loss = 0

			torch.cuda.empty_cache()

		epoch_loss = sum(iter_loss_total) / len(iter_loss_total)
		# epoch_loss_train.append(float(epoch_loss))
		wrtLoss(epoch, float(epoch_loss), save_path, timeSince(start_train))
		writer.add_scalar('TrainData Loss',epoch_loss,epoch)

		print("Epoch: {}; Percent complete: {:.1f}%; Average loss: {:.4f}; Time: {} / {}" 
					.format(epoch, epoch * 100 / num_epochs, epoch_loss, timeSince(start_epoch), timeSince(start_train)))

		# Save checkpoint
		if (epoch % 5 == 0) & (float(epoch_loss)<=3):
			torch.save({
				'en': encoder.state_dict(),
				'en_opt': encoder_optimizer.state_dict(),
				'de': decoder.state_dict(),
				'de_opt': decoder_optimizer.state_dict(),
			}, os.path.join(save_path, 'E{}_{}.tar'.format(str(epoch), 'checkpoint')))


	# print("Writing loss ...")
	# wrtLoss(epoch_loss_train, epoch_loss_eval, save_path, timeSince(start_train))


def jieba_train(data_mode, data, encoder, decoder, attn_model, teacher_forcing_ratio=0, loss=0, criterion=None, max_len=0):

	if data[2] is None: # no ans
		sent_tensors, qus_tensors = [d.to(device) for d in data if d is not None]
		anspos_tensor = None
	else:
		sent_tensors, qus_tensors, anspos_tensor = [d.to(device) for d in data]

	encoder_hidden = encoder.initHidden(len(sent_tensors[0]))
	encoder_outputs, encoder_hidden = encoder(sent_tensors, encoder_hidden, anspos_tensor)
	last_hidden_fwd = encoder_hidden[0]
	last_hidden_bwd = encoder_hidden[1]
	encoder_hidden = last_hidden_fwd + last_hidden_bwd
	encoder_hidden = torch.unsqueeze(encoder_hidden, 0)

	loss, all_tokens = decoder_part(data_mode, qus_tensors, encoder_hidden, encoder_outputs, max_len, attn_model, 
						encoder, decoder, teacher_forcing_ratio=teacher_forcing_ratio, loss=loss, criterion=criterion)

	return loss, all_tokens

def bert_train(data_mode, data, encoder, decoder, attn_model, teacher_forcing_ratio=0, loss=0, criterion=None, max_len=0):

	sent_tensors, sent_segtensors, sent_masktensors, qus_tensors = [d.to(device) for d in data]

	context_hidden = encoder(sent_tensors, sent_segtensors, sent_masktensors)	
	context_hidden = torch.transpose(context_hidden,0,1).contiguous()

	loss, all_tokens = decoder_part(data_mode, qus_tensors, context_hidden, context_hidden, max_len, attn_model, 
					encoder, decoder, teacher_forcing_ratio=teacher_forcing_ratio, loss=loss, criterion=criterion)

	return loss, all_tokens


def decoder_part(data_mode, qus_tensors, encoder_hidden, encoder_outputs, max_len, attn_model, encoder, decoder, teacher_forcing_ratio, loss, criterion):

	decoder_input = torch.LongTensor([[qus_tensors[0][idx] for idx in range(len(qus_tensors[0]))]])
	decoder_input = decoder_input.to(device)

	decoder_hidden = encoder_hidden[:decoder.n_layers]

	if max_len==0:
		decoder_len = len(qus_tensors)
	else:
		decoder_len = max_len

	# Initialize tensors to append decoded words to
	all_tokens = torch.zeros([0], device=device, dtype=torch.long)

	# Forward batch of sequences through decoder one time step at a time
	for t in range(1, decoder_len):

		if attn_model == 'none':
			decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
		else:
			decoder_output, decoder_hidden, decoder_attention = decoder(
				decoder_input, decoder_hidden, encoder_outputs)


		use_teacher_forcing = True if random.random() <= teacher_forcing_ratio else False

		# Teacher forcing: next input is current target
		if use_teacher_forcing:
			decoder_input = qus_tensors[t].view(1, -1)
			all_tokens = torch.cat((all_tokens, decoder_input), dim=0)

		# No teacher forcing: next input is decoder's own current output
		else:
			_, topi = decoder_output.topk(1)
			decoder_input = torch.LongTensor([[topi[i][0] for i in range(len(qus_tensors[0]))]])
			decoder_input = decoder_input.to(device)
			all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
				
		if criterion!= None:
			# Calculate and accumulate loss
			loss += criterion(decoder_output, qus_tensors[t])
		
	# torch.cuda.empty_cache()

	return loss, all_tokens


def train_evaluation(data_mode, data_sel, dev_loader, attn_model, vocab, tokenizer, encoder, decoder, criterion, save_path, epoch):

	# Initialize variables
	iter_loss = 0.0
	iter_loss_avg = 0.0

	for it, data in enumerate(dev_loader):

		if data_mode == "jieba":
			loss, all_tokens = jieba_train(data_mode, data, encoder, decoder, attn_model, criterion=criterion)
			iter_loss += float(loss / len(data[0]))

		elif data_mode == "bert":
			loss, all_tokens = bert_train(data_mode, data, encoder, decoder, attn_model, criterion=criterion)
			iter_loss += float(loss / len(data[0][0]))

	iter_loss_avg = float(iter_loss / it)

	f = open(save_path +"/train_devds.txt", 'a', encoding = 'utf-8', errors='ignore')
	f.write("=== Epoch "+str(epoch)+" === loss: "+str(iter_loss_avg)+" \n")

	print_data(f, data_mode, data_sel, data, vocab, tokenizer, all_tokens)

	f.close()

	return iter_loss_avg


def print_data(f, data_mode, data_sel, data, vocab, tokenizer, all_tokens):

	if data_mode == "jieba":
		if data[2] is None: # no ans
			sent_tensors, qus_tensors = [d.to(device) for d in data if d is not None]
			anspos_tensor = None
		else:
			sent_tensors, qus_tensors, anspos_tensor = [d.to(device) for d in data]
		b_size = len(data[0][0])
	elif data_mode =="bert":
		sent_tensors, _, _, qus_tensors = [d.to(device) for d in data]
		b_size = len(data[0])

	sumB0 = []
	sumB1 = []
	sumB2 = []
	sumB3 = []
	sumB4 = []
	sumR0 = []
	sumR1 = []
	sumR2 = []

	for row in range(b_size):
		# write s
		if data_mode == "jieba":
			tokens = [vocab.index2word[int(idx)] for idx in sent_tensors[:,row]]
			tokens[:] = [word for word in takewhile(lambda x: (x!='PAD') & (x!='EOS'), tokens)]
			tokens = tokens[1:]
			f.write("> ")
			f.write(' '.join(tokens)+"\n")

		elif data_mode == "bert":
			tokens = tokenizer.convert_ids_to_tokens(sent_tensors[row,:].tolist())
			tokens[:] = [word for word in takewhile(lambda x: x!='[SEP]', tokens)]
			tokens = tokens[1:]
			f.write("> ")
			f.write(''.join(tokens)+"\n")

		# write q
		if data_sel=="sq":
			if data_mode == "jieba":
				tokens = [vocab.index2word[int(idx)] for idx in qus_tensors[:,row]]
				tokens[:] = [word for word in takewhile(lambda x: x!='EOS', tokens)]
				tokens = tokens[1:]
				f.write("< ")
				f.write(' '.join(tokens)+"\n")
			elif data_mode == "bert":
				tokens = tokenizer.convert_ids_to_tokens(qus_tensors[:,row].tolist())
				tokens[:] = [word for word in takewhile(lambda x: x!='[SEP]', tokens)]
				tokens = tokens[1:]
				f.write("< ")
				f.write(''.join(tokens)+"\n")

		# write output
		if data_mode == "jieba":
			evaldata_tokens = all_tokens[:,row].tolist()
			output_words = [vocab.index2word[token] for token in evaldata_tokens]
			# output_words = tokenizer.convert_ids_to_tokens(evaldata_tokens)
			output_words[:] = [word for word in takewhile(lambda x: (x!='EOS'), output_words)]
			f.write("= ")
			f.write(' '.join(output_words)+"\n")
		elif data_mode == "bert":
			evaldata_tokens = all_tokens[:,row].tolist()
			output_words = tokenizer.convert_ids_to_tokens(evaldata_tokens)
			output_words[:] = [word for word in takewhile(lambda x: x!='[SEP]', output_words)]
			f.write("= ")
			f.write(''.join(output_words)+"\n")

		reference_b = [list(''.join(tokens).replace("UNK","$"))]
		candidate_b = list(''.join(output_words).replace("UNK","$"))

		bleu0 = round(sentence_bleu(reference_b, candidate_b), 5)
		bleu1 = round(sentence_bleu(reference_b, candidate_b, weights=(1, 0, 0, 0)), 5)
		bleu2 = round(sentence_bleu(reference_b, candidate_b, weights=(1, 1, 0, 0)), 5)
		bleu3 = round(sentence_bleu(reference_b, candidate_b, weights=(1, 1, 1, 0)), 5)
		bleu4 = round(sentence_bleu(reference_b, candidate_b, weights=(1, 1, 1, 1)), 5)

		f.write("[ BLEU: "+str(bleu0)+" ] ( "+str(bleu1)+
		", "+str(bleu2)+", "+str(bleu3)+", "+str(bleu4)+" )\n")

		sumB0 += [bleu0]
		sumB1 += [bleu1]
		sumB2 += [bleu2]
		sumB3 += [bleu3]
		sumB4 += [bleu4]

		reference_r = ' '.join(list(''.join(tokens).replace("UNK","$")))
		candidate_r = ' '.join(list(''.join(output_words).replace("UNK","$")))

		rouge = Rouge()
		try:
			scores = rouge.get_scores(candidate_r, reference_r)
			rouge0 = round(scores[0]["rouge-l"]['f'], 5)
			rouge1 = round(scores[0]["rouge-1"]['r'], 5)
			rouge2 = round(scores[0]["rouge-2"]['r'], 5)

			f.write("[ ROUGE: "+str(rouge0)+" ] ( "+str(rouge1)+", "+str(rouge2)+" )\n\n")

			sumR0 += [rouge0]
			sumR1 += [rouge1]
			sumR2 += [rouge2]

		except ValueError:
			f.write("[ ROUGE: None ]\n\n")


	return sumB0, sumB1, sumB2, sumB3, sumB4, sumR0, sumR1, sumR2