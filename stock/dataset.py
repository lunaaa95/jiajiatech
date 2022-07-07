import torch
from torch.utils.data import Dataset
import numpy as np
import pickle as pkl
import random
import ipdb


LOWER = torch.tensor([-1, -0.02, 0.0, 0.02, 0.05], dtype=torch.float)
UPPER = torch.tensor([-0.02, 0.0, 0.02, 0.05, 1], dtype=torch.float)


def value_to_soft_onehot(value, epson: float):
    # softly split value into right continuous intervals
    soft_onehot = epson/(len(UPPER)-1) * torch.ones(len(LOWER), dtype=torch.float)
    soft_onehot[(LOWER < value) & (value <= UPPER)] = 1.0 - epson
    return soft_onehot


def value_to_onehot(value):
    # split value into right continuous intervals
    onehot = torch.zeros(len(LOWER), dtype=torch.float)
    onehot[(LOWER < value) & (value <= UPPER)] = 1.0
    return onehot


def value_to_class(value):
    # split value into right continuous interval class
    onehot = torch.zeros(len(LOWER), dtype=torch.int)
    onehot[LOWER < value] = 1
    return onehot.sum()


class StockData(Dataset):
    def __init__(self, data, memory_length, smooth_factor, mask_on, identifiable=False):
        self.data = data
        self.memory_length = memory_length
        self.smooth_factor = smooth_factor
        self.mask_on = mask_on
        self.identifiable = identifiable

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        observation = torch.tensor(self.data[idx][0], dtype=torch.float)
        smooth_target = value_to_soft_onehot(self.data[idx][1], self.smooth_factor)
        target = value_to_onehot(self.data[idx][1])
        ticker, date = self.data[idx][2:]
        mask = torch.zeros(self.memory_length, dtype=torch.bool)
        if self.mask_on:
            mask[idx] = True
        if self.identifiable:
            return observation, smooth_target, target, mask, ticker, date
        else:
            return observation, smooth_target, target, mask


def split_data(path_to_pickle, smooth_factor, cut_num, independent_mem=True, mem_ratio=0.5, val_ratio=0.1):
    print('start split data')
    with open(path_to_pickle, 'rb') as f:
        content = pkl.load(f, encoding='latin1')
    thrs = int(len(content) / cut_num)

    # split memory and training data
    memory = []
    train = []
    if independent_mem:
        # memory data is separated from training data
        assert 1 - mem_ratio - val_ratio > 0, 'no data for training'
        for i in range(cut_num):
            memory.extend(content[i*thrs: i*thrs+int(mem_ratio*thrs)])
            train.extend(content[i*thrs+int(mem_ratio*thrs): i*thrs+int((1-val_ratio)*thrs)])
        memory = random.sample(memory, 30000)
    else:
        # memory data is integrated with training data
        assert (1 - val_ratio) > 0, 'no data for training'
        for i in range(cut_num):
            train.extend(content[i*thrs: i*thrs+int((1-val_ratio)*thrs)])
        memory = train
    n_mem = len(memory)
    memory = {
        'keys': torch.stack([torch.tensor(x[0], dtype=torch.float) for x in memory]),
        'values': torch.stack([value_to_onehot(x[1]) for x in memory]),
        'tickers': [x[2] for x in memory],
        'dates': [x[3] for x in memory],
    }
    # memory = (key, memory)
    # memory key: (N, seq_len, input_size)
    # memory value: (N, value_size)
    train_dataset = StockData(
        data=train,
        memory_length=n_mem,
        smooth_factor=smooth_factor,
        mask_on=not independent_mem,
        identifiable=False
    )

    # split validating data
    val = []
    for i in range(cut_num):
        val.extend(content[i*thrs+int(0.9*thrs): (i+1)*thrs])
    val_dataset = StockData(
        data=val,
        memory_length=n_mem,
        smooth_factor=smooth_factor,
        mask_on=False,
        identifiable=True
        )

    print(f'full length of data: {len(content)}')
    return memory, train_dataset, val_dataset
