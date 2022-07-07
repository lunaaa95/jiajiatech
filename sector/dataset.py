import torch
from torch.utils.data import Dataset
import numpy as np
import pickle as pkl
import random
import ipdb

LOWER = torch.tensor([[-10, -0.01, -0.003, 0.001, 0.013],
                      [-10, -0.01, -0.002, 0.0025, 0.009],
                      [0, 0.0125, 0.02, 0.04, 0.08]], dtype=torch.float)
UPPER = torch.tensor([[-0.01, -0.003, 0.001, 0.013, 10],
                      [-0.01, -0.002, 0.0025, 0.009, 10],
                      [0.0125, 0.02, 0.04, 0.08, 1]], dtype=torch.float)


def value_to_soft_onehot(value, tgt_class, epson: float):
    value = value[tgt_class]
    # softly split value into right continuous intervals
    soft_onehot = epson / (len(UPPER[tgt_class]) - 1) * torch.ones(len(LOWER[tgt_class]), dtype=torch.float)
    soft_onehot[(LOWER[tgt_class] < value) & (value <= UPPER[tgt_class])] = 1.0 - epson
    return soft_onehot


def value_to_onehot(value, tgt_class):
    value = value[tgt_class]
    # split value into right continuous intervals
    onehot = torch.zeros(len(LOWER[tgt_class]), dtype=torch.float)
    onehot[(LOWER[tgt_class] < value) & (value <= UPPER[tgt_class])] = 1.0
    return onehot


def value_to_class(value, tgt_class):
    value = value[tgt_class]
    # split value into right continuous interval class
    onehot = torch.zeros(len(LOWER[tgt_class]), dtype=torch.int)
    onehot[LOWER[tgt_class] < value] = 1
    return onehot.sum()


class SectorData(Dataset):
    def __init__(self, data, tgt_class, memory_length, smooth_factor, mask_on, identifiable=False):
        self.data = data
        self.tgt_class = tgt_class
        self.memory_length = memory_length
        self.smooth_factor = smooth_factor
        self.mask_on = mask_on
        self.identifiable = identifiable

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        observation = torch.tensor(self.data[idx][0], dtype=torch.float)
        smooth_target = value_to_soft_onehot(self.data[idx][1], self.tgt_class, self.smooth_factor)
        target = value_to_onehot(self.data[idx][1], self.tgt_class)
        ticker, date = self.data[idx][2:]
        mask = torch.zeros(self.memory_length, dtype=torch.bool)
        if self.mask_on:
            mask[idx] = True
        if self.identifiable:
            return observation, smooth_target, target, mask, ticker, date
        else:
            return observation, smooth_target, target, mask


def split_data(path_to_pickle, smooth_factor, cut_num, tgt_class, independent_mem=True, mem_ratio=0.5, val_ratio=0.1):
    assert -1 < tgt_class < 3, 'tgt_class should be 0, 1 or 2'
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
            memory.extend(content[i * thrs: i * thrs + int(mem_ratio * thrs)])
            train.extend(content[i * thrs + int(mem_ratio * thrs): i * thrs + int((1 - val_ratio) * thrs)])
        memory = random.sample(memory, 10000)
    else:
        # memory data is integrated with training data
        assert (1 - val_ratio) > 0, 'no data for training'
        for i in range(cut_num):
            train.extend(content[i * thrs: i * thrs + int((1 - val_ratio) * thrs)])
        memory = train
    n_mem = len(memory)
    memory = {
        'keys': torch.stack([torch.tensor(x[0], dtype=torch.float) for x in memory]),
        'values': torch.stack([value_to_onehot(x[1], tgt_class) for x in memory]),
        'sectors': [x[2] for x in memory],
        'dates': [x[3] for x in memory],
    }
    # memory = (key, memory)
    # memory key: (N, seq_len, input_size)
    # memory value: (N, value_size)
    train_dataset = SectorData(
        data=train,
        tgt_class=tgt_class,
        memory_length=n_mem,
        smooth_factor=smooth_factor,
        mask_on=not independent_mem,
        identifiable=False
    )

    # split validating data
    val = []
    for i in range(cut_num):
        val.extend(content[i * thrs + int((1 - val_ratio) * thrs): (i + 1) * thrs])
    val_dataset = SectorData(
        data=val,
        tgt_class=tgt_class,
        memory_length=n_mem,
        smooth_factor=smooth_factor,
        mask_on=False,
        identifiable=True
    )

    print(f'full length of data: {len(content)}')
    return memory, train_dataset, val_dataset
