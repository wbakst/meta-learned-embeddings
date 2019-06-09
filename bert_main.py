import argparse

from bert_lstm import BiLSTM
from bert_maml import MetaLearner
from data_loader import ReviewDataset
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
## TENSORBOARD LOGGING ##
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()

parser.add_argument('--use_gpu', action='store_true')
parser.add_argument('--num_epochs', type=int, default=10000)
parser.add_argument('--batch_size', type=int, default=10) # num tasks per batch
parser.add_argument('--K', type=int, default=5) # K-shot learning
parser.add_argument('--num_classes', type=int, default=2)

parser.add_argument('--meta_lr', type=float, default=1e-3)
parser.add_argument('--update_lr', type=float, default=0.1)
parser.add_argument('--num_updates', type=int, default=1)

parser.add_argument('--max_length', type=int, default=512)
parser.add_argument('--vocab_size', type=int, default=138398)
parser.add_argument('--embedding_size', type=int, default=768)
parser.add_argument('--hidden_size', type=int, default=768)
parser.add_argument("--output_dir",
                    default='./maml_output',
                    type=str,
                    required=False,
                    help="The output directory where tensorboard output will be written.")

args = parser.parse_args()

model = BiLSTM(args)

# model.load_state_dict(torch.load('./weights.pt'))
# model.eval()

### INIT TB LOGGING ###
save_dir = args.output_dir + "/few_shot/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
tbx = SummaryWriter(save_dir)

meta_learner = MetaLearner(model, args)

train_dataset = ReviewDataset('train', args)
dev_dataset = ReviewDataset('dev', args)
# test_dataset = ReviewDataset('test', args)
print('train tasks: ', train_dataset.num_tasks)
print('dev tasks: ', dev_dataset.num_tasks)
# print('test tasks:', test_dataset.num_tasks)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
dev_loader = DataLoader(dev_dataset, batch_size=dev_dataset.num_tasks)
# test_loader = DataLoader(test_dataset, batch_size=test_dataset.num_tasks)

lowest_dev_loss = float('inf')

# sum_avgs = []

for epoch in range(1, args.num_epochs+1):
    num_tensorboard_steps = 0
    print('EPOCH %d' % epoch)
    print('TRAIN (batch size = %d)' % args.batch_size)
    for batch_idx, batch in enumerate(train_loader):
        num_tensorboard_steps += 1
        # sample without replacement from same task for train and test (K of each)
        train_x, train_y, train_lens, test_x, test_y, test_lens = batch

        # train the metalearner
        losses, accs = meta_learner.forward(train_x, train_y, train_lens, test_x, test_y, test_lens, tbx, num_tensorboard_steps, evaluate=False)
        print(losses, accs)

    print('DEV (batch size = %d)' % dev_dataset.num_tasks)
    for batch_idx, batch in enumerate(dev_loader):
        num_tensorboard_steps += 1
        train_x, train_y, train_lens, test_x, test_y, test_lens = batch

        losses, accs = meta_learner.forward(train_x, train_y, train_lens, test_x, test_y, test_lens, tbx, num_tensorboard_steps, evaluate=True)
        if losses[1] < lowest_dev_loss:
            lowest_dev_loss = losses[1]
            torch.save(model.state_dict(), './bert-maml-weights.pt')
        print(losses, accs)
        # sum_avgs.append(accs[1])

    # print(np.sum(sum_avgs) / epoch)

    # print('TEST (batch size = %d)' % test_dataset.num_tasks)
    # for batch_idx, batch in enumerate(test_loader):

    #     train_x, train_y, train_lens, test_x, test_y, test_lens = batch

    #     losses, accs = meta_learner.forward(train_x, train_y, train_lens, test_x, test_y, test_lens, evaluate=True)
    #     print(losses, accs)
