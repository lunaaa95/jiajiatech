import torch
import numpy
import pickle
from torch.utils.data import Dataset, random_split
import ipdb


class IndustryTestData(Dataset):
    def __init__(self, content, seq_len, device):
        super().__init__()
        self.content = content
        self.seq_len = seq_len
        self.device = device

    def __len__(self):
        return len(self.content) - self.seq_len
    def __getitem__(self, idx):
        obs_data = self.content[idx: idx+self.seq_len]
        tgt_data = self.content[idx+self.seq_len]
        obs = torch.stack([torch.tensor(x[1], dtype=torch.float) for x in obs_data]).contiguous().to(self.device)
        tgt = torch.tensor(tgt_data[1], dtype=torch.float, device=self.device)
        date = tgt_data[0]
        return date, obs, tgt


class IndustryData(Dataset):
    def __init__(self, data):
        super(IndustryData, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data[2])

    def __getitem__(self, idx):
        obs = self.data[0][idx]
        tgt = self.data[1][idx]
        date = self.data[2][idx]
        return date, obs, tgt


class IndustryDataSets:
    def __init__(self, path, seq_len, n_sgm, train_ratio, device):
        with open(path, 'rb') as f:
            self.content = pickle.load(f)
        assert seq_len < len(self.content), 'seq_len is too long'
        self.seq_len = seq_len
        self.n_sgm = n_sgm
        self.train_ratio = train_ratio
        self.device = device
        self.train_data = [[], [], []]
        self.test_data = [[], [], []]
        self._partition()

    def _partition(self):
        sgm_len = int(len(self) / self.n_sgm)
        n_train = int(sgm_len * self.train_ratio)
        for i in range(self.n_sgm):
            sgm_start = sgm_len*i
            sgm_end = min(sgm_len*(i+1)+self.seq_len, len(self))
            train_data = self.content[sgm_start: sgm_start+self.seq_len+n_train]
            # length of each train segment: n_train + seq_len
            test_data = self.content[sgm_start+n_train: sgm_end]
            # length of each test segment: n_test + seq_len
            for j in range(n_train):
                obs_data = train_data[j: j + self.seq_len]
                tgt_data = train_data[j + self.seq_len]
                self.train_data[0].append(torch.stack([torch.tensor(x[1], dtype=torch.float) for x in obs_data]))
                self.train_data[1].append(torch.tensor(tgt_data[1], dtype=torch.float))
                self.train_data[2].append(tgt_data[0])
            for j in range(sgm_end - sgm_start - n_train - self.seq_len):
                obs_data = test_data[j: j + self.seq_len]
                tgt_data = test_data[j + self.seq_len]
                self.test_data[0].append(torch.stack([torch.tensor(x[1], dtype=torch.float) for x in obs_data]))
                self.test_data[1].append(torch.tensor(tgt_data[1], dtype=torch.float))
                self.test_data[2].append(tgt_data[0])
        self.train_data[0] = torch.stack(self.train_data[0]).contiguous().to(self.device)
        self.train_data[1] = torch.stack(self.train_data[1]).contiguous().to(self.device)
        self.test_data[0] = torch.stack(self.test_data[0]).contiguous().to(self.device)
        self.test_data[1] = torch.stack(self.test_data[1]).contiguous().to(self.device)

    def __len__(self):
        return len(self.content) - self.seq_len

    @property
    def input_size(self):
        return self.content[0][1].shape[-1]

    def split_data(self):
        train_dataset = IndustryData(self.train_data)
        test_dataset = IndustryData(self.test_data)
        return train_dataset, test_dataset
