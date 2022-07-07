import torch
from tqdm import tqdm
import random
import ipdb

ATPP, TPP, PT, PP = 0, 1, 2, 3


def cross_entropy_loss(prediction: torch.Tensor, target: torch.Tensor):
    log_probs = torch.log(prediction)
    loss = (-1 * torch.sum(target * log_probs, dim=1)).mean()
    return loss


def train_epoch(dataloader, memory, model, loss_fn, optimizer, evaluate, device):
    num_batches = len(dataloader.dataset)
    acc = 0.0
    total_loss = 0.0
    model.train()
    for query, smooth_target, target, mask in tqdm(dataloader):
        model.zero_grad()
        prediction = model(memory, query.to(device), mask.to(device))
        loss = loss_fn(prediction, smooth_target.to(device))
        loss.backward()
        acc += evaluate(prediction, target.to(device))
        optimizer.step()
        total_loss += loss.item()
    acc /= num_batches
    print(f"Loss during Training: {total_loss:>8f}")
    print(f"Avg Accuracy during Training: {acc:>8f}")
    return acc


def test_epoch(dataloader, memory, model, evaluate, device):
    num_batches = len(dataloader.dataset)
    eval_metric = torch.zeros(4, memory[1].shape[1], device=device)
    model.eval()
    with torch.no_grad():
        for query, smooth_target, target, mask, ticker, date in dataloader:
            prediction = model(memory, query.to(device), mask.to(device))
            atpp, tpp, pt, pp = evaluate(prediction, target.to(device))
            eval_metric[ATPP] += atpp
            eval_metric[TPP] += tpp
            eval_metric[PT] += pt
            eval_metric[PP] += pp
    print('Accuracy:', eval_metric[ATPP] / num_batches)
    print('Recall:', eval_metric[TPP] / eval_metric[PT])
    print('Precision', eval_metric[TPP] / eval_metric[PP])
    return eval_metric[ATPP][0] / num_batches


def predict(dataloader, memory, model, device):
    model.eval()
    predictions = []
    attns = []
    targets = []
    tickers = []
    dates = []
    with torch.no_grad():
        for query, smooth_target, target, mask, ticker, date in dataloader:
            predictions.append(model(memory, query.to(device), mask.to(device)))
            attns.append(model.get_attention(memory, query.to(device), mask.to(device)))
            targets.append(target)
            tickers.extend(ticker)
            dates.extend(date)
    predictions = torch.vstack(predictions).cpu()
    attns = torch.vstack(attns).cpu()
    targets = torch.vstack(targets).cpu()
    return predictions, targets, tickers, dates, attns
