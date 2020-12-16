import torch, os
from statistics import mean 
from itertools import takewhile
from _train import jieba_train
from _train import bert_train
from _train import print_data

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

def train_evaluation(data_mode, data_sel, dev_loader, attn_model, vocab, tokenizer, encoder, decoder, save_dir, checkpoint_epoch):
	save_path = os.path.join(save_dir)
	if not os.path.exists(save_path):
		os.makedirs(save_path)

	# Initialize variables
	iter_loss = 0.0
	iter_loss_avg = 0.0
	criterion = torch.nn.NLLLoss(ignore_index=0)

	for it, data in enumerate(dev_loader):

		if data_mode == "jieba":
			loss, all_tokens = jieba_train(data_mode, data, encoder, decoder, attn_model, criterion=criterion)
			iter_loss += float(loss / len(data[0]))

		elif data_mode == "bert":
			loss, all_tokens = bert_train(data_mode, data, encoder, decoder, attn_model, criterion=criterion)
			iter_loss += float(loss / len(data[0][0]))

	iter_loss_avg = float(iter_loss / it)

	f = open(save_path +"/train_devds.txt", 'a', encoding = 'utf-8', errors='ignore')
	f.write("=== Epoch "+str(checkpoint_epoch)+" === loss: "+str(iter_loss_avg)+" \n")

	print_data(f, data_mode, data_sel, data, vocab, tokenizer, all_tokens)

	f.close()


def test_evaluation(eval_flag, data_mode, data_sel, eval_loader, attn_model, vocab, tokenizer,
					encoder, decoder, max_length, save_dir, checkpoint_epoch):
	save_path = os.path.join(save_dir)
	if not os.path.exists(save_path):
		os.makedirs(save_path)

	if eval_flag==True:
		print("Writing eval data ...")
		f = open(save_path + "/eval_testds_E"+str(checkpoint_epoch)+".txt", 'a', encoding = 'utf-8', errors='ignore')
	else:
		print("Writing dev data ...")
		f = open(save_path + "/eval_devds_E"+str(checkpoint_epoch)+".txt", 'a', encoding = 'utf-8', errors='ignore')

	avgB0 = []
	avgB1 = []
	avgB2 = []
	avgB3 = []
	avgB4 = []
	avgR0 = []
	avgR1 = []
	avgR2 = []

	for _, data in enumerate(eval_loader):
		if data_mode == "jieba":
			_, all_tokens = jieba_train(data_mode, data, encoder, decoder, attn_model, max_len=max_length)
			
		elif data_mode == "bert":

			_, all_tokens = bert_train(data_mode, data, encoder, decoder, attn_model, max_len=0)

		sumB0, sumB1, sumB2, sumB3, sumB4, sumR0, sumR1, sumR2 = print_data(f, data_mode, data_sel, data, vocab, tokenizer, all_tokens)

		avgB0 += sumB0
		avgB1 += sumB1
		avgB2 += sumB2
		avgB3 += sumB3
		avgB4 += sumB4
		avgR0 += sumR0
		avgR1 += sumR1
		avgR2 += sumR2


	f.write("[ avgBLEU: "+str(mean(avgB0))+" ]\n( "+str(mean(avgB1))+
		", "+str(mean(avgB2))+", "+str(mean(avgB3))+", "+str(mean(avgB4))+" )\n")
	f.write("[ avgROUGE: "+str(mean(avgR0))+" ] ( "+str(mean(avgR1))+", "+str(mean(avgR2))+" )\n")


	f.close()