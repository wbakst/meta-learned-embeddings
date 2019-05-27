import numpy as np
from torch.utils.data.dataset import Dataset
import os
import torch

class ReviewDataset(Dataset):
    def __init__(self, split, args):

        self.data_dir = 'preprocessed_data'

        self.K = args.K
        self.num_classes = args.num_classes
        self.max_length = args.max_length
            
        self.tasks = []
        for file in os.listdir(self.data_dir):
            if file == '.DS_Store':
                continue
            if split not in file:
                continue
            pos_examples, neg_examples = self.read_file(file)
            if len(pos_examples) < self.K or len(neg_examples) < self.K:
                continue # skip for now if not enough examples
            self.tasks.append((pos_examples, neg_examples))

        self.num_tasks = len(self.tasks)

    def read_file(self, file):
        pos_examples = []
        neg_examples = []
        # filename = '{}.{}.{}'.format(filepath, label_type, split)
        with open(os.path.join(self.data_dir, file), 'r') as file:
            for line in file:
                x = [0 for _ in range(self.max_length)]
                line = line.split()
                length = min(len(line[:-1]), self.max_length)
                for i in range(length):
                    x[i] = int(line[i])
                # x = [int(wid) for wid in line[:-1]]
                y = int((int(line[-1])+1)/2)
                if y == 0:
                    neg_examples.append((x, y, length))
                else:
                    pos_examples.append((x, y, length))
        return np.array(pos_examples), np.array(neg_examples)

    def __getitem__(self, index):
        # choose the task indicated by index
        pos_examples, neg_examples = self.tasks[index]

        # for now just choose randomly among examples
        pos_indices = np.random.choice(range(len(pos_examples)), size=self.K)
        neg_indices = np.random.choice(range(len(neg_examples)), size=self.K)

        pos = pos_examples[pos_indices]
        neg = neg_examples[neg_indices]

        # interleave randomly
        ex = np.empty((self.K*2, 3), dtype=pos.dtype)
        if np.random.uniform() > .5:
            ex[0::2,:] = pos
            ex[1::2,:] = neg
        else:
            ex[0::2,:] = neg
            ex[1::2,:] = pos

        train_ex = ex[:self.K]
        test_ex = ex[self.K:]

        train_data = torch.tensor(np.stack(train_ex[:,0]))
        train_labels = torch.tensor(np.stack(train_ex[:,1]))
        train_lens = torch.tensor(np.stack(train_ex[:,2]))

        test_data = torch.tensor(np.stack(train_ex[:,0]))
        test_labels = torch.tensor(np.stack(train_ex[:,1]))
        test_lens = torch.tensor(np.stack(train_ex[:,2]))

        return train_data, train_labels, train_lens, test_data, test_labels, test_lens

    def __len__(self):
        return len(self.tasks)
