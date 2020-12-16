import os, time, math
import torch, numpy, pandas
from gensim.models import KeyedVectors
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class Vocabulary(object):

	def __init__(self, data_path, min_count):
		self.index2word = {0: 'PAD', 1: 'SOS', 2: 'EOS', 3: 'UNK'}
		self.word2index = {'PAD': 0, 'SOS': 1, 'EOS': 2, 'UNK': 3}
		self.word2count = {}
		self.num_words = 4  # Count SOS, EOS, PAD, UNK

		print("Building Vocabulary ...")
		self.build_vocab(data_path+"_seg.csv")
		self.trim(min_count)

	def build_vocab(self, data_path):
		with open(data_path, 'r', encoding='utf-8') as dataset:
			for line in dataset.readlines():
				# line for content, question, answer
				line = line.replace("\n","").split(" , ")
				self.addSentence(line[0]) # content
				self.addSentence(line[1]) # question
				# self.addSentence(line[2]) # answer

	def addSentence(self, sentence):
		words = sentence.split(" ")
		for w in words:
			self.addWord(w)

	def addWord(self, word):
		if word not in self.word2index:
			self.word2index[word] = self.num_words
			self.word2count[word] = 1
			self.index2word[self.num_words] = word
			self.num_words += 1
		else:
			self.word2count[word] += 1

	def trim(self, min_count):
		keep_words = []
		for k, v in self.word2count.items():
			if (v>=min_count):
				keep_words.append(k)
				
		# print('keep_words {} / {} = {:.4f}'.format(len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)))

		# Reinitialize dictionaries
		self.index2word = {0: 'PAD', 1: 'SOS', 2: 'EOS', 3: 'UNK'}
		self.word2index = {'PAD': 0, 'SOS': 1, 'EOS': 2, 'UNK': 3}
		self.word2count = {}
		self.num_words = 4  # Count SOS, EOS, PAD, UNK
		
		for word in keep_words:
			self.addWord(word)


def build_emb(save_dir, vec_path, vocab, emb_size, loadEmbedding):

	save_path = os.path.join(save_dir)
	if not os.path.exists(save_path):
		os.makedirs(save_path)

	print("Setting Embedding ...")
	embedding = torch.nn.Embedding(vocab.num_words, emb_size)
	if loadEmbedding:
		emb_matrix = torch.load(loadEmbedding)
		embedding_sd = emb_matrix['embedding']
		embedding.load_state_dict(embedding_sd)
	else:
		pretrain_words = KeyedVectors.load_word2vec_format(vec_path, binary=False)
		embedding = emb_weight(pretrain_words, vocab, emb_size)
		torch.save({'embedding': embedding.state_dict()}, os.path.join(save_path, 'emb_matrix.tar'))
	
	return embedding

def emb_weight(pretrain_words, vocab, emb_size):

	weights_matrix = numpy.zeros((vocab.num_words, emb_size))
	for index, word in vocab.index2word.items():
		if(word == 'PAD'):
			weights_matrix[index] = numpy.zeros(emb_size)
		elif(word == 'SOS'):
			weights_matrix[index] = numpy.ones(emb_size)
		elif(word == 'EOS'):
			weights_matrix[index] = numpy.full(emb_size, vocab.word2index["EOS"])
		elif(word == 'UNK'):
			weights_matrix[index] = numpy.full(emb_size, vocab.word2index["UNK"])
		else:
			try: 
				weights_matrix[index] = pretrain_words[word]
			except KeyError as msg:
				weights_matrix[index] = numpy.random.uniform(low=-1, high=1, size=(emb_size))

	weight_tensor = torch.FloatTensor(weights_matrix)
	return torch.nn.Embedding.from_pretrained(weight_tensor)


class DRCDDataset(Dataset):
	
	def __init__(self, dataset, data, mode, with_ans, vocab, tokenizer):
		# assert dataset in ["train", "dev", "test"]
		assert data in ["ss", "qq", "sq"]
		assert mode in ["jieba", "bert"]

		# self.dataset = dataset
		self.data = data
		self.mode = mode
		self.with_ans = with_ans
		if self.mode == "jieba":
			self.df = pandas.read_csv(dataset+"_seg.csv", sep=" , ", engine= 'python')
		elif self.mode == "bert":
			self.df = pandas.read_csv(dataset+".csv", sep=",")
		self.len = len(self.df)
		self.vocab = vocab
		self.tokenizer = tokenizer

	# 定義回傳一筆訓練 / 測試數據的函式
	def __getitem__(self, idx):

		if self.data == "ss":
			sent = self.df.iloc[idx, 0]
			qus = self.df.iloc[idx, 0]
			ans = None
		elif self.data == "qq":
			sent = self.df.iloc[idx, 1]
			qus = self.df.iloc[idx, 1]
			ans = None
		elif self.data == "sq":
			sent = self.df.iloc[idx, 0]
			qus = self.df.iloc[idx, 1]
			ans = self.df.iloc[idx, 2]
		
		if self.mode == "jieba":
			sent_tensor = self.seq2idx(sent)
			qus_tensor = self.seq2idx(qus)

			if self.with_ans:
				if ans == None:
					ans_tensor = torch.zeros([0], dtype=torch.long)
					anspos_tensor = None
				else:
					ans_tensor = self.seq2idx(ans, False, False)
					anspos_tensor = self.anspos2idx(ans_tensor.tolist(), sent_tensor.tolist())

				if anspos_tensor is None:
					anspos_tensor = torch.zeros_like(sent_tensor)
				
				anspos_tensor = anspos_tensor.type(torch.LongTensor)
			else:
				anspos_tensor = None
			
			return (sent_tensor, qus_tensor, anspos_tensor)

		elif self.mode == "bert":
			# 建立 BERT tokens
			word_pieces = ["[CLS]"]
			tokens_sent = self.tokenizer.tokenize(sent)
			word_pieces += tokens_sent
			word_pieces += ["[SEP]"]	
			# 將整個 token 序列轉換成索引序列
			sent_ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
			sent_tensor = torch.tensor(sent_ids)
			# 將token 位置設為 0
			sent_segtensor = torch.zeros_like(sent_tensor)

			# qus_tensor = self.seq2idx(qus) # for fasttext output
			word_pieces = ["[CLS]"]
			tokens_qus = self.tokenizer.tokenize(qus)
			word_pieces += tokens_qus
			word_pieces += ["[SEP]"]
			qus_ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
			qus_tensor = torch.tensor(qus_ids)

			if self.with_ans:
				if ans == None:
					ans_tensor = torch.zeros([0], dtype=torch.long)
					anspos_tensor = None
				else:
					tokens_ans = self.tokenizer.tokenize(ans)
					ans_ids = self.tokenizer.convert_tokens_to_ids(tokens_ans)
					ans_tensor = torch.tensor(ans_ids)
					anspos_tensor = self.anspos2idx(ans_tensor.tolist(), sent_tensor.tolist())

				if anspos_tensor == None:
					anspos_tensor = torch.zeros_like(sent_tensor)

				sent_segtensor = anspos_tensor.type(torch.LongTensor)
			
			return (sent_tensor, sent_segtensor, qus_tensor)

	
	def __len__(self):
		return self.len

	def seq2idx(self, sent, add_eos=True, add_sos=True):
		idx_sent = [self.vocab.word2index["SOS"]] if add_sos else []
		sent = sent.split(" ")
		for word in sent:
			if word not in self.vocab.word2index:
				idx_sent.append((self.vocab.word2index["UNK"]))
			else:
				idx_sent.append(self.vocab.word2index[word])
		if add_eos:
			idx_sent.append(self.vocab.word2index["EOS"])
		sent_tensor = torch.tensor(idx_sent)
		return sent_tensor

	def anspos2idx(self, ans, sent):
		for idx in (i for i,tk in enumerate(sent) if tk==ans[0]):
			if sent[idx:idx+len(ans)]==ans:
				# so_and = idx, eo_ans = idx+len(ans)-1
				pos_tensor = torch.zeros(len(sent))
				for ans_l in range(len(ans)):
					pos_tensor[int(idx)+ans_l] = 1
				return pos_tensor


def create_bert_batch(samples):

	sent_tensors = [s[0] for s in samples]
	sent_tensors = pad_sequence(sent_tensors, batch_first=True)
	
	sent_segtensors = [s[1] for s in samples]
	sent_segtensors = pad_sequence(sent_segtensors, batch_first=True)

	# attention masks，將 sent_tensors 裡頭不為 zero padding
	# 的位置設為 1 讓 BERT 只關注這些位置的 tokens
	sent_masktensors = torch.zeros(sent_tensors.shape, dtype=torch.long)
	sent_masktensors = sent_masktensors.masked_fill(sent_tensors != 0, 1)

	qus_tensors = [s[2] for s in samples]
	qus_tensors = pad_sequence(qus_tensors, batch_first=False) # for jieba
	
	return sent_tensors, sent_segtensors, sent_masktensors, qus_tensors

def create_jieba_batch(samples):

	# seperate source and target sequences
	sent_tensors = [s[0] for s in samples]
	sent_tensors = pad_sequence(sent_tensors)
	qus_tensors = [s[1] for s in samples]
	qus_tensors = pad_sequence(qus_tensors)
	if samples[0][2] is not None:
		ans_tensors = [s[2] for s in samples]
		ans_tensors = pad_sequence(ans_tensors)
		ans_tensors = ans_tensors.type(torch.FloatTensor)
	else:
		ans_tensors = None

	return sent_tensors, qus_tensors, ans_tensors
