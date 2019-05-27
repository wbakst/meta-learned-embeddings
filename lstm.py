import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BiLSTM(nn.Module):

    def __init__(self, args):

        super(BiLSTM, self).__init__()

        self.embedding_size = args.embedding_size
        self.hidden_size = args.hidden_size
        self.vocab_size = args.vocab_size #args.vocab_size
        self.label_size = args.num_classes
        self.batch_size = args.K # batch size is per task here
        self.use_gpu = args.use_gpu

        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_size)

        self.lstm = nn.LSTM(input_size=self.embedding_size,
                            hidden_size=self.hidden_size,
                            bidirectional=True,
                            batch_first=True)

        self.hidden = self.init_hidden()

        self.classifier = nn.Linear(self.hidden_size*2, self.label_size)

    def init_hidden(self):
        h_0 = Variable(torch.zeros(2, self.batch_size, self.hidden_size), requires_grad=True)
        c_0 = Variable(torch.zeros(2, self.batch_size, self.hidden_size), requires_grad=True)
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
        last_output = output.gather(1, last_indices).squeeze(1)
        # print(last_output.size())
        logits = self.classifier(last_output)
        return logits
