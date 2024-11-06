import torch
import json
from torch.utils.data import Dataset


class KnownsDataset(Dataset):
    def __init__(self, data_pth='./data/known_1000.json'):
        with open(data_pth, 'r') as f:
            self.dataset = json.load(f)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.dataset[index]
    

if __name__ == '__main__':
    dataset = KnownsDataset()

    for i in range(10):
        print(dataset[i])