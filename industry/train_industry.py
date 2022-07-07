import ipdb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import random
import pickle, datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from dataset import IndustryDataSets, IndustryTestData
from model import GRU
from config import get_args
from evaluate import evaluate, evaluate_rank, predict_top

# hold the same random seed
random.seed(443132)
np.random.seed(443132)
top_rank = 5
bottom = 2

if __name__ == '__main__':
    args = get_args()
    print('model parameters'.upper())
    for k, v in vars(args).items():
        print(k.ljust(20), '->', v)
    print('-' * 51)
    device = 'cuda:' + str(args.cuda)

    datasets = IndustryDataSets(
        path=args.input,
        seq_len=args.seq_len,
        n_sgm=args.sgm,
        train_ratio=args.train_ratio,
        device=device
    )
    train_dataset, val_dataset = datasets.split_data()
    with open(args.test_input, 'rb') as f:
        test_content = pickle.load(f)
    test_dataset = IndustryTestData(
        content=test_content,
        seq_len=args.seq_len,
        device=device
    )

    dataloader_train = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    dataloader_val = DataLoader(
        val_dataset,
        batch_size=200,
        shuffle=False,
    )
    dataloader_test = DataLoader(
        test_dataset,
        batch_size=200,
        shuffle=False,
    )
    input_size = datasets.input_size

    model_GRU = GRU(
        input_size=input_size,
        nlayers=1,
        nhid=args.hidden_size,
        dropout=0
    ).cuda(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model_GRU.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)

    training_losses = []
    val_losses = []
    prediction_hits = np.array([]).reshape(0, top_rank)
    score = []
    overall_score = []
    for epoch in range(args.num_epochs):
        print("epoch", epoch)
        # train
        model_GRU.train()
        hidden = model_GRU.init_hidden(args.batch_size)
        for i, (date, observation, target) in enumerate(dataloader_train):
            hidden = hidden.detach()
            model_GRU.zero_grad()
            decoded = model_GRU(observation, None)
            # decoded: bsz, input_size
            # target: bsz, input_size
            loss = loss_fn(decoded, target)
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print("iter", i, "loss", loss.item())
        training_losses.append(loss.item())

        # validate
        total_confidence = 0
        hit_counts = np.zeros(top_rank)
        target_amount = 0
        model_GRU.eval()
        for i, (date, observation, target) in enumerate(dataloader_val):
            prediction = model_GRU(observation, None)
            # softmax = nn.Softmax(dim=1)
            # probability = softmax(prediction)

            top_hit_sum = evaluate_rank(prediction, target, top_rank)
            for hit in range(0, top_rank):
                hit_counts[hit] += top_hit_sum[top_hit_sum == hit+1].count_nonzero().item()
            target_amount += target.shape[0]

            # confidence = torch.sum(probability * target)
            # total_confidence += (confidence / data.shape[0])

        prediction_hits = np.append(prediction_hits, (hit_counts / target_amount).reshape(1, top_rank), axis=0)
        # print("confidence:", total_confidence.item() / (i+1))
        print("overall hit ratio:{:.3f}".format((hit_counts*np.arange(1, top_rank+1).reshape(1, top_rank)).sum()
                                                / top_rank / target_amount))
        for hit in range(0, top_rank):
            print('{:d} hit ratio:{:.3f}'.format(hit+1, hit_counts[hit] / target_amount), end='|')
        print('')
        print('-' * 51)

    final_prediction = []
    final_dates = []
    model_GRU.eval()
    with torch.no_grad():
        for date, observation, target in dataloader_val:
            final_prediction.append(model_GRU(observation, None))
            final_dates.extend(date)
    final_prediction = torch.vstack(final_prediction)

    # ipdb.set_trace()
    test_prediction = []
    test_dates = []
    model_GRU.eval()
    with torch.no_grad():
        for date, observation, target in dataloader_test:
            test_prediction.append(model_GRU(observation, None))
            test_dates.extend(date)
    test_prediction = torch.vstack(test_prediction)

    now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_file = args.output + '_' + now
    test_output_file = args.test_output + '_' + now
    with open(output_file, 'wb') as f:
        pickle.dump((final_dates, final_prediction.cpu()), f)
    with open(test_output_file, 'wb') as f:
        pickle.dump((test_dates, test_prediction.cpu()), f)
    print(f'{now} finished')
