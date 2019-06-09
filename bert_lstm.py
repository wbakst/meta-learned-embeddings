import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, param_obj):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(param_obj['vocab_size'], param_obj['hidden_size'], padding_idx=0)
        self.position_embeddings = nn.Embedding(param_obj['max_position_embeddings'], param_obj['hidden_size'])
        self.token_type_embeddings = nn.Embedding(param_obj['type_vocab_size'], param_obj['hidden_size'])

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(param_obj['hidden_size'], eps=param_obj['layer_norm_eps'])
        self.dropout = nn.Dropout(param_obj['hidden_dropout_prob'])

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BiLSTM(nn.Module):

    def __init__(self, args):

        super(BiLSTM, self).__init__()

        self.embedding_size = args.embedding_size
        self.hidden_size = args.hidden_size
        self.vocab_size = args.vocab_size #args.vocab_size
        self.label_size = args.num_classes
        self.batch_size = args.K # batch size is per task here
        self.use_gpu = args.use_gpu

        param_obj = {
            'vocab_size': self.vocab_size,
            'hidden_size': self.hidden_size,
            'max_position_embeddings':512,
            'type_vocab_size':2,
            'layer_norm_eps': 1e-12,
            'hidden_dropout_prob': 0.1
        }

        self.embeddings = BertEmbeddings(param_obj)
        print("param_obj:",param_obj)
        #self.embeddings = nn.Embedding(self.vocab_size, self.embedding_size)

        self.lstm = nn.LSTM(input_size=self.embedding_size,
                            hidden_size=self.hidden_size,
                            bidirectional=True,
                            batch_first=True)

        self.hidden = self.init_hidden(self.use_gpu)

        self.classifier = nn.Linear(self.hidden_size*2, self.label_size)

	if self.use_gpu:
	    self.embeddings = self.embeddings.cuda()
	    self.lstm = self.lstm.cuda()
	    self.classifier = self.classifier.cuda()

    def init_hidden(self, use_gpu):
        h_0 = Variable(torch.zeros(2, self.batch_size, self.hidden_size), requires_grad=True)
        c_0 = Variable(torch.zeros(2, self.batch_size, self.hidden_size), requires_grad=True)
	if use_gpu:
	    return h_0.cuda(), c_0.cuda()
	else:
            return h_0, c_0

    def forward(self, word_ids, lengths):
        # sort by length
        lengths, perm_idx = lengths.sort(0, descending=True)
        word_ids = word_ids[perm_idx]

        # print('word ids', word_ids.size())
        # word_ids = word_ids.permute(1, 0)
        embs = self.embeddings(word_ids)#.view(word_ids.size(1), self.batch_size, -1) # maybe permute instead
        # print('embs', embs.size())
        # print('lengths', lengths.size())
        packed = pack_padded_sequence(embs, lengths, batch_first=True)
        # print('packed', packed.size())
        # embs = embs.permute(1, 0, 2)
        # print(packed.size())
        output, self.hidden = self.lstm(packed, self.hidden)
        output, _ = pad_packed_sequence(output, batch_first=True)

        # unsort
        _, unperm_idx = perm_idx.sort(0)
        output = output[unperm_idx]
        lengths = lengths[unperm_idx]

        # get final output state
        last_indices = (lengths - 1).view(-1, 1).expand(len(lengths), output.size(2)).unsqueeze(1) # 1 = time dimension
	if self.use_gpu:
	    last_indices = last_indices.cuda()
        last_output = output.gather(1, last_indices).squeeze(1)
        # print(last_output.size())
        logits = self.classifier(last_output)
        return logits
