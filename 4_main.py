import os, torch, argparse
from torch import optim
from torch.utils.data import DataLoader
from pytorch_pretrained_bert import BertTokenizer

from _preprocessing import Vocabulary, DRCDDataset
from _preprocessing import build_emb, create_bert_batch, create_jieba_batch
from _model import EncoderRNN, BertEncoder
from _model import DecoderRNN, LuongAttnDecoderRNN
from _train import trainEpochs
from _evaluation import train_evaluation, test_evaluation

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


def main(args):

	print("Data preprocessing ...")
	if args.data_mode == "jieba":
		vocab = Vocabulary(args.data_set, args.vec_min)
		tokenizer = None
		data_transformer = DRCDDataset(args.data_set, args.data_sel, args.data_mode, args.with_ans, vocab, tokenizer)
		data_loader = DataLoader(data_transformer, batch_size=args.batch_size, shuffle=True, collate_fn=create_jieba_batch)
		embedding = build_emb(args.save_dir, args.vec_path, vocab, args.emb_size, args.loadEmbedding)
		embedding = embedding.to(device)
	elif args.data_mode == "bert":
		vocab = None
		tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
		data_transformer = DRCDDataset(args.data_set, args.data_sel, args.data_mode, args.with_ans, vocab, tokenizer)
		data_loader = DataLoader(data_transformer, batch_size=args.batch_size, shuffle=True, collate_fn=create_bert_batch)
		embedding = None

	print('Building encoder and decoder ...')
	if args.data_mode == "jieba":
		encoder = EncoderRNN(embedding, args.hidden_size, args.transfer_layer, args.encoder_n_layers, args.dropout)
		vocab_size = vocab.num_words
	elif args.data_mode == "bert":
		encoder = BertEncoder(args.transfer_layer)
		embedding = encoder.embedding
		vocab_size = encoder.vocab_size

	if args.attn_model == 'none':
		decoder = DecoderRNN(embedding, args.hidden_size, vocab_size, args.decoder_n_layers, args.dropout)
	else:
		decoder = LuongAttnDecoderRNN(args.attn_model, embedding, args.hidden_size, vocab_size, args.decoder_n_layers, args.dropout)


	# Load model if a loadFilename is provided
	if args.loadEncoder:
		print("Loading pretrained Encoder ...")
		checkpoint = torch.load(args.loadEncoder)
		prencoder_sd = checkpoint['en']

		encoder_sd = encoder.state_dict()
		prencoder_sd = {k: v for k, v in encoder_sd.items() if k in prencoder_sd}
		encoder_sd.update(prencoder_sd) 
		encoder.load_state_dict(encoder_sd)

		if args.fixed_enc:
			for param in encoder.parameters():
				param.requires_grad = False
			encoder.out.weight.requires_grad = True
			encoder.out.bias.requires_grad = True

	if args.loadDecoder:
		print("Loading pretrained Decoder ...")
		checkpoint = torch.load(args.loadDecoder)
		decoder_sd = checkpoint['de']
		decoder.load_state_dict(decoder_sd)
	
	if args.loadFilename:
		print("Loading pretrained Model ...")
		checkpoint = torch.load(args.loadFilename)
		encoder_sd = checkpoint['en']
		encoder.load_state_dict(encoder_sd)
		decoder_sd = checkpoint['de']
		decoder.load_state_dict(decoder_sd)

	# Use appropriate device
	encoder = encoder.to(device)
	decoder = decoder.to(device)	
	# Ensure dropout layers are in train mode
	encoder.train()
	decoder.train()

	if args.training_flag:
		print('Building optimizers ...')
		if args.fixed_enc:
			encoder_optimizer = optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()), lr=args.encoder_op_lr)
		else:
			encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.encoder_op_lr)

		decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.decoder_op_lr)

		if args.loadEncoder:
			checkpoint = torch.load(args.loadEncoder)
			prencoder_optimizer_sd = checkpoint['en_opt']

			encoder_optimizer_sd = encoder_optimizer.state_dict()
			prencoder_optimizer_sd = {k: v for k, v in encoder_optimizer_sd.items() if k in prencoder_optimizer_sd}
			encoder_optimizer_sd.update(prencoder_optimizer_sd) 
			encoder_optimizer.load_state_dict(encoder_optimizer_sd)

		if args.loadDecoder:
			checkpoint = torch.load(args.loadDecoder)
			decoder_optimizer_sd = checkpoint['de_opt']
			decoder_optimizer.load_state_dict(decoder_optimizer_sd)

		if args.loadFilename:
			checkpoint = torch.load(args.loadFilename)
			prencoder_optimizer_sd = checkpoint['en_opt']

			encoder_optimizer_sd = encoder_optimizer.state_dict()
			prencoder_optimizer_sd = {k: v for k, v in encoder_optimizer_sd.items() if k in prencoder_optimizer_sd}
			encoder_optimizer_sd.update(prencoder_optimizer_sd) 
			encoder_optimizer.load_state_dict(encoder_optimizer_sd)

			decoder_optimizer_sd = checkpoint['de_opt']
			decoder_optimizer.load_state_dict(decoder_optimizer_sd)

		# If you have cuda, configure cuda to call
		for state in encoder_optimizer.state.values():
			for k, v in state.items():
				if isinstance(v, torch.Tensor):
					state[k] = v.cuda()
		for state in decoder_optimizer.state.values():
			for k, v in state.items():
				if isinstance(v, torch.Tensor):
					state[k] = v.cuda()

		print("Starting training!")
		trainEpochs(args.save_dir, args.data_mode, data_loader, vocab, tokenizer, 
			args.attn_model, encoder, decoder, encoder_optimizer, decoder_optimizer, args.it_percent,
			args.checkpoint_epoch, args.num_epochs, args.teacher_forcing_ratio, args.data_sel)


	# Set dropout layers to eval mode
	encoder.eval()
	decoder.eval()

	if args.dev_flag:
		dev_transformer = DRCDDataset(args.dev_set, args.data_sel, args.data_mode, args.with_ans, vocab, tokenizer)
		if args.data_mode == "jieba":
			dev_loader = DataLoader(dev_transformer, batch_size=args.batch_size, shuffle=True, collate_fn=create_jieba_batch)
		elif args.data_mode == "bert":
			dev_loader = DataLoader(dev_transformer, batch_size=args.batch_size, shuffle=True , collate_fn=create_bert_batch)

		print("Starting evaluation!")
		test_evaluation(args.eval_flag, args.data_mode, args.data_sel, dev_loader, args.attn_model, vocab, tokenizer, 
						encoder, decoder, args.max_length, args.save_dir, args.checkpoint_epoch)
	
	if args.eval_flag:
		eval_transformer = DRCDDataset(args.eval_set, args.data_sel, args.data_mode, args.with_ans, vocab, tokenizer)
		if args.data_mode == "jieba":
			eval_loader = DataLoader(eval_transformer, batch_size=args.batch_size, shuffle=False, collate_fn=create_jieba_batch)
		elif args.data_mode == "bert":
			eval_loader = DataLoader(eval_transformer, batch_size=args.batch_size, shuffle=False, collate_fn=create_bert_batch)

		print("Starting evaluation!")
		test_evaluation(args.eval_flag, args.data_mode, args.data_sel, eval_loader, args.attn_model, vocab, tokenizer, 
						encoder, decoder, args.max_length, args.save_dir, args.checkpoint_epoch)


if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	# for data
	parser.add_argument('--vec_min', type=int, default=5)
	parser.add_argument('--vec_path', type=str, default='cc.zh.300.vec') #cc.zh.300.vec
	parser.add_argument('--emb_size', type=int, default=300)
	parser.add_argument('--batch_size', type=int, default=32)
	
	parser.add_argument('--data_set', type=str, default="sS200_train")
	parser.add_argument('--dev_set', type=str, default="sS200_dev")
	parser.add_argument('--data_sel', type=str, default="sq")
	parser.add_argument('--data_mode', type=str, default="bert")

	# for model
	parser.add_argument('--model_name', type=str, default='BERT_model') # ignore
	parser.add_argument('--attn_model', type=str, default='general') #dot #general #concat
	parser.add_argument('--with_ans', type=bool, default=False)
	parser.add_argument('--fixed_enc', type=bool, default=False)
	parser.add_argument('--transfer_layer', type=bool, default=False)

	parser.add_argument('--hidden_size', type=int, default=300)
	parser.add_argument('--encoder_n_layers', type=int, default=1)
	parser.add_argument('--decoder_n_layers', type=int, default=1)
	parser.add_argument('--dropout', type=float, default=0.1)

	# for training
	parser.add_argument('--training_flag', type=bool, default=True)
	parser.add_argument('--it_percent', type=int, default=0.5)
	parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5)
	parser.add_argument('--encoder_op_lr', type=float, default=1e-4)
	parser.add_argument('--decoder_op_lr', type=float, default=1e-4)
	parser.add_argument('--num_epochs', type=int, default=300)
	
	# for loading
	save_dir = 'BERT_SQ'
	check_epoch = 0
	encoder_epoch = 0
	decoder_epoch = 0
	loadFilename = os.path.join(save_dir, 'E{}_checkpoint.tar'.format(check_epoch))
	loadEncodername = os.path.join(save_dir, 'SS_E{}_checkpoint.tar'.format(encoder_epoch))
	loadDecodername = os.path.join(save_dir, 'QQ_E{}_checkpoint.tar'.format(decoder_epoch))
	loadEmbedding = os.path.join(save_dir, 'emb_matrix.tar')
	if not os.path.exists(loadFilename):
		loadFilename = None
	if not os.path.exists(loadEncodername):
		loadEncodername = None
	if not os.path.exists(loadDecodername):
		loadDecodername = None
	if not os.path.exists(loadEmbedding):
		loadEmbedding = None

	parser.add_argument('--loadFilename', default=loadFilename)
	parser.add_argument('--loadEncoder', default=loadEncodername)
	parser.add_argument('--loadDecoder', default=loadDecodername)
	parser.add_argument('--loadEmbedding', default=loadEmbedding)
	parser.add_argument('--save_dir', type=str, default=save_dir)
	parser.add_argument('--checkpoint_epoch', type=int, default=check_epoch)

	# for evaluation
	parser.add_argument('--dev_flag', type=bool, default=False)
	parser.add_argument('--eval_flag', type=bool, default=False)
	parser.add_argument('--eval_set', type=str, default="sQ30_test")
	parser.add_argument('--max_length', type=int, default=50)

	args = parser.parse_args(args=[])

	print("[["+args.model_name+"]]")

	main(args)
