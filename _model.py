import torch
import torch.nn.functional as F
from pytorch_pretrained_bert import BertModel, BertSelfAttention


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

class EncoderRNN(torch.nn.Module):
	def __init__(self, embedding, hidden_size, transfer_layer, n_layers=1, dropout=0):
		super(EncoderRNN, self).__init__()
		self.hidden_size = hidden_size
		self.dropout = dropout
		self.transfer_layer = transfer_layer

		self.embedding = embedding
		self.embedding_dropout = torch.nn.Dropout(dropout)
		# Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
		#   because our input size is a word embedding with number of features == hidden_size
		self.gru = torch.nn.GRU(hidden_size, hidden_size, bidirectional=True)

		if transfer_layer:
			self.out = torch.nn.Linear(hidden_size, hidden_size)

	def forward(self, input_sent, hidden, ans_pos):
		# Convert word indexes to embeddings
		embedded = self.embedding(input_sent)
		embedded = self.embedding_dropout(embedded)
		if ans_pos is not None:
			ans_pos = torch.unsqueeze(ans_pos, 2)
			embedded = torch.cat((embedded, ans_pos), dim=2)
		outputs, hidden = self.gru(embedded, hidden)
		# Sum bidirectional GRU outputs
		outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
		if self.transfer_layer:
			outputs = self.out(outputs)		
		# Return output and final hidden state
		return outputs, hidden

	def initHidden(self, batch_size):
		return torch.zeros(2, batch_size, self.hidden_size, device=device)

class DecoderRNN(torch.nn.Module):
	def __init__(self, embedding, hidden_size, output_size, n_layers=1, dropout=0):
		super(DecoderRNN, self).__init__()

		# Keep for reference
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.n_layers = n_layers
		self.dropout = dropout

		# Define layers
		self.embedding = embedding
		self.embedding_dropout = torch.nn.Dropout(dropout)
		# self.trans_in = torch.nn.Linear(768, hidden_size)
		self.gru = torch.nn.GRU(hidden_size, hidden_size)
		self.out = torch.nn.Linear(hidden_size, output_size)
		self.softmax = torch.nn.LogSoftmax(dim=1)

	def forward(self, input_step, last_hidden):
		# Note: we run this one step (word) at a time
		# Get embedding of current input word
		embedded = self.embedding(input_step)
		embedded = self.embedding_dropout(embedded)
		# Forward through unidirectional GRU
		# if len(last_hidden[0][0]) == 768:
		# 	input_hidden = self.trans_in(last_hidden)
		# 	rnn_output, hidden = self.gru(embedded, input_hidden)
		# else:
		rnn_output, hidden = self.gru(embedded, last_hidden)
		# Predict next word using Luong eq. 6
		output = self.out(rnn_output[0])
		output = self.softmax(output)
		# Return output and final hidden state
		return output, hidden

class Attn(torch.nn.Module):
	def __init__(self, method, hidden_size):
		super(Attn, self).__init__()
		self.method = method
		if self.method not in ['dot', 'general', 'concat']:
			raise ValueError(self.method, "is not an appropriate attention method.")
		self.hidden_size = hidden_size
		if self.method == 'general':
			self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
		elif self.method == 'concat':
			self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
			self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

	def dot_score(self, hidden, encoder_output):
		return torch.sum(hidden * encoder_output, dim=2)

	def general_score(self, hidden, encoder_output):
		energy = self.attn(encoder_output)
		return torch.sum(hidden * energy, dim=2)

	def concat_score(self, hidden, encoder_output):
		energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
		return torch.sum(self.v * energy, dim=2)

	def forward(self, hidden, encoder_outputs):
		# Calculate the attention weights (energies) based on the given method
		if self.method == 'general':
			attn_energies = self.general_score(hidden, encoder_outputs)
		elif self.method == 'concat':
			attn_energies = self.concat_score(hidden, encoder_outputs)
		elif self.method == 'dot':
			attn_energies = self.dot_score(hidden, encoder_outputs)

		# Transpose max_length and batch_size dimensions
		attn_energies = attn_energies.t()

		# Return the softmax normalized probability scores (with added dimension)
		return F.softmax(attn_energies, dim=1).unsqueeze(1)

class LuongAttnDecoderRNN(torch.nn.Module):
	def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0):
		super(LuongAttnDecoderRNN, self).__init__()

		# Keep for reference
		self.attn_model = attn_model
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.n_layers = n_layers
		self.dropout = dropout

		# Define layers
		self.embedding = embedding
		self.embedding_dropout = torch.nn.Dropout(dropout)
		self.gru = torch.nn.GRU(hidden_size, hidden_size)
		self.concat = torch.nn.Linear(hidden_size * 2, hidden_size)
		self.out = torch.nn.Linear(hidden_size, output_size)
		self.softmax = torch.nn.LogSoftmax(dim=1)

		self.attn = Attn(attn_model, hidden_size)

	def forward(self, input_step, last_hidden, encoder_outputs):
		# Note: we run this one step (word) at a time
		if self.hidden_size==768:
			# Forward through unidirectional GRU
			with torch.no_grad():
				embedded = self.embedding(input_step)
			embedded = self.embedding_dropout(embedded)
			# Forward through unidirectional GRU
			rnn_output, hidden = self.gru(embedded, last_hidden)
		else:
			# Get embedding of current input word
			embedded = self.embedding(input_step)
			embedded = self.embedding_dropout(embedded)
			if self.hidden_size == 301:
				ans_pos = torch.unsqueeze(torch.zeros_like(input_step, dtype=torch.float, device=device), 2)
				embedded = torch.cat((embedded, ans_pos), dim=2)
			# Forward through unidirectional GRU
			rnn_output, hidden = self.gru(embedded, last_hidden)

		# Calculate attention weights from the current GRU output
		attn_weights = self.attn(rnn_output, encoder_outputs)
		# Multiply attention weights to encoder outputs to get new "weighted sum" context vector
		context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
		# Concatenate weighted context vector and GRU output using Luong eq. 5
		rnn_output = rnn_output.squeeze(0)
		context = context.squeeze(1)
		concat_input = torch.cat((rnn_output, context), 1)
		concat_output = torch.tanh(self.concat(concat_input))
		# Predict next word using Luong eq. 6
		output = self.out(concat_output)
		# output = F.softmax(output, dim=1)
		output = self.softmax(output)
		# Return output and final hidden state
		return output, hidden, attn_weights

class BertEncoder(torch.nn.Module):
	def __init__(self, output_flag):
		super(BertEncoder, self).__init__()

		self.bert_model, self.config = BertModel.from_pretrained("bert-base-chinese")
		self.embedding = self.bert_model.embeddings.word_embeddings
		self.bert_selfAttn = BertSelfAttention(self.config)
		self.vocab_size = self.config.vocab_size
		self.output_flag = output_flag

		if output_flag:
			self.out = torch.nn.Linear(self.config.hidden_size, self.config.hidden_size)

	def forward(self, sent_tensors, seg_tensors, masks_tensors):

		encoder_hidden, _, extended_attention_mask = self.bert_model(input_ids=sent_tensors, token_type_ids=seg_tensors, 
										attention_mask=masks_tensors, output_all_encoded_layers=False)

		context_hidden = self.bert_selfAttn(encoder_hidden, extended_attention_mask)	

		if self.output_flag:
			encoder_hidden = self.out(encoder_hidden)	

		return encoder_hidden, context_hidden
		
