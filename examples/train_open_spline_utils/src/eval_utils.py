
import numpy as np
import torch


def to_one_hot(target, maxx=50):
    target = torch.from_numpy(target.astype(np.int64)).cuda()
    N = target.shape[0]
    target_one_hot = torch.zeros((N, maxx))

    target_one_hot = target_one_hot.cuda()
    target_t = target.unsqueeze(1)
    target_one_hot = target_one_hot.scatter_(1, target_t.long(), 1)
    return target_one_hot
