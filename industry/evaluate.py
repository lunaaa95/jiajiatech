import torch
import numpy as np


class Evaluate:
    pass


def evaluate(pred, ground_true):
    pred = pred.detach()
    ground_true = ground_true.detach()
    pred = torch.argsort(pred, dim=1)
    pred_onehot = torch.zeros((ground_true.shape[0], ground_true.shape[1]), dtype=int)
    n_list = []
    for row in range(pred.shape[0]):
        max_n = int(torch.sum(ground_true[row]).item())
        max_index = [i.item() for i in pred[row, -max_n:]]
        pred_onehot[row, max_index] = 1
        n_list.append(max_n)

    # n_list targe标签有几个1
    bingo = torch.sum(pred_onehot * ground_true, dim=1)

    return (sum(bingo / np.array(n_list)) / len(bingo)).item()
    # return pred_onehot
    # return sum(bingo)


def predict_top(prediction, top_rank):
    predicted_rank = prediction.argsort(dim=1, descending=True)
    predicted_top = torch.zeros(prediction.shape, dtype=torch.int)
    columns = predicted_rank[:, top_rank - 1].reshape(prediction.shape[0])
    rows = torch.tensor(range(0, prediction.shape[0])).reshape(prediction.shape[0])
    predicted_threshold = prediction[rows, columns].reshape((prediction.shape[0], 1))
    predicted_top[prediction >= predicted_threshold] = 1
    return predicted_top


def evaluate_rank(prediction, ground_truth, top_rank):
    true_threshold = (top_rank + 0.5)/top_rank/(top_rank+1)
    true_top = torch.zeros(ground_truth.shape, dtype=torch.int)
    true_top[ground_truth > true_threshold] = 1
    return (predict_top(prediction, top_rank) * true_top).sum(dim=1)


# def describe(data):
#     for hit in range(0, top_rank):
#     hit_counts[hit] += top_hit_sum[top_hit_sum == hit+1].count_nonzero().item()
# target_amount += target.shape[0]