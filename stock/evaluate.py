import torch


def calc_acc(prediction: torch.Tensor, target: torch.Tensor):
    rp = prediction.argsort(dim=1, descending=True)[:, 0]
    rt = target.argsort(dim=1, descending=True)[:, 0]
    atpp = (rp == rt).sum()
    # atpp: all true positive predictions
    return atpp


def evaluate_by_class(prediction: torch.Tensor, target: torch.Tensor):
    """
    Calculate recall and precision by each class

    Args:
        prediction: (batch_size, value_size)
        target: (batch_size, value_size)

    Returns: sum of true positive predictions,
    sum of positive targets,
    sum of positive predictions,
    shapes are all (1, value_size)
    """
    rp = prediction.argsort(dim=1, descending=True)[:, 0]
    # rp: rankings in prediction
    rt = target.argsort(dim=1, descending=True)[:, 0]
    # rt: rankings in target
    tpp = rt[rp == rt]
    # tpp: true positive predictions
    atpp = (rp == rt).sum()
    # atpp: all true positive predictions
    sum_pt = target.sum(dim=0)
    # sum_pt: sum of positive targets
    sum_tpp = torch.zeros(sum_pt.size(), device=sum_pt.device)
    # sum_tpp: sum of true positive predictions
    sum_pp = torch.zeros(sum_pt.size(), device=sum_pt.device)
    # sum_pp: sum of positive predictions
    for i in range(sum_pt.size(0)):
        sum_tpp[i] = torch.sum(tpp == i)
        sum_pp[i] = torch.sum(rp == i)
    return atpp, sum_tpp, sum_pt, sum_pp
