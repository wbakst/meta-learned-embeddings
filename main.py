import argparse

from lstm import BiLSTM
from maml import MetaLearner
from data_loader import ReviewDataset
import numpy as np
import os
import torch
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()

parser.add_argument('--use_gpu', action='store_true')
parser.add_argument('--num_epochs', type=int, default=10000)
parser.add_argument('--batch_size', type=int, default=8) # num tasks per batch
parser.add_argument('--K', type=int, default=5) # K-shot learning
parser.add_argument('--num_classes', type=int, default=2)

parser.add_argument('--meta_lr', type=float, default=1e-3)
parser.add_argument('--update_lr', type=float, default=0.1)
parser.add_argument('--num_updates', type=int, default=1)

parser.add_argument('--max_length', type=int, default=100)
parser.add_argument('--vocab_size', type=int, default=138398)
parser.add_argument('--embedding_size', type=int, default=50)
parser.add_argument('--hidden_size', type=int, default=100)

args = parser.parse_args()

model = BiLSTM(args)
meta_learner = MetaLearner(model, args)
dataset = ReviewDataset(args)
batch_size = args.batch_size

# load twice the batch size then split into train/test
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

for epoch in range(args.num_epochs):

    for batch_idx, batch in enumerate(data_loader):

        # sample without replacement from same task for train and test (K of each)
        train_x, train_y, train_lens, test_x, test_y, test_lens = batch

        # train the metalearner
        losses, accs = meta_learner.forward(train_x, train_y, train_lens, test_x, test_y, test_lens)
        print(losses, accs)
