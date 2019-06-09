import numpy as np
from torch.utils.data.dataset import Dataset
import os
import torch

class ReviewDataset(Dataset):
    def __init__(self, split, args):

        self.data_dir = 'preprocessed_data'

        self.use_gpu = args.use_gpu

        self.K = args.K
        self.num_classes = args.num_classes
        self.max_length = args.max_length

        # given in Diverse Few-Shot Text Classification with Multiple Metrics paper
        test_cats = ['books', 'dvd', 'electronics', 'kitchen_housewares']

        # chosen by me
        dev_cats = ['apparel', 'camera_photo', 'magazines', 'office_products']

        # note: we're ignoring train/dev/test in the file names and separating
        # by product category instead

        self.tasks = {}
        for file in sorted(os.listdir(self.data_dir)):
            if file == '.DS_Store':
                continue
            cat, cutoff, _ = file.split('.')
            if split == 'train' and (cat in dev_cats or cat in test_cats):
                continue
            if split == 'dev' and cat not in dev_cats:
                continue
            if split == 'test' and cat not in test_cats:
                continue
            pos_examples, neg_examples = self.read_file(file)
            task = cat+'.'+cutoff
            if task not in self.tasks:
                self.tasks[task] = ([], [])
            self.tasks[task][0].extend(pos_examples)
            self.tasks[task][1].extend(neg_examples)

        task_list = []
        task_names = []
        for task in self.tasks:
            pos_examples = self.tasks[task][0]
            neg_examples = self.tasks[task][1]
            if len(pos_examples) < self.K or len(neg_examples) < self.K:
                print('not enough examples', task)
                continue # skip for now if not enough examples
            task_list.append((np.array(pos_examples), np.array(neg_examples)))
            task_names.append(task)

        self.tasks = task_list
        self.task_names = task_names

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
        return pos_examples, neg_examples

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

        if self.use_gpu:
            train_data = train_data.cuda()
            train_labels = train_labels.cuda()
            train_lens = train_lens.cuda()

            test_data = test_data.cuda()
            test_labels = test_labels.cuda()
            train_lens = train_lens.cuda()

        return train_data, train_labels, train_lens, test_data, test_labels, test_lens

    def __len__(self):
        return len(self.tasks)
