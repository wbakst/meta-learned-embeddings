import numpy as np
from torch.utils.data.dataset import Dataset

class ReviewDataset(Dataset):
    def __init__(self, filepath, label_type, split):
    	self.data, self.labels = [], []
        filename = '{}.{}.{}'.format(filepath, label_type, split)
        with open(filename, 'r') as file:
        	for line in file:
        		line = line.split()
        		self.data.append([int(wid) for wid in line[:-1]])
        		self.labels.append(int(line[-1]))
        self.data, self.labels = np.array(self.data), np.array(self.labels)
        
    def __getitem__(self, index):
        # stuff
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)