import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import scipy.sparse as sp


def loss_corr(x, nnodes=None):
    if nnodes is None:
        nnodes = x.shape[0]
    idx = np.random.choice(x.shape[0], int(np.sqrt(nnodes)))
    x = x[idx]
    x = x - x.mean(0)
    cov = x.t() @ x
    I_k = torch.eye(x.shape[1]).cuda() / np.sqrt(x.shape[1])
    loss = torch.norm(cov/torch.norm(cov) - I_k)
    return loss


def torch_corr(x):
    mean_x = torch.mean(x, 1)
    # xm = x.sub(mean_x.expand_as(x))
    xm = x - mean_x.view(-1, 1)
    c = xm.mm(xm.t())
    # c = c / (x.size(1) - 1)

    # normalize covariance matrix
    d = torch.diag(c)
    stddev = torch.pow(d, 0.5)
    c = c.div(stddev.expand_as(c))
    c = c.div(stddev.expand_as(c).t())

    # clamp between -1 and 1
    # probably not necessary but numpy does it
    c = torch.clamp(c, -1.0, 1.0)
    return c


def get_pairwise_sim(x):
    try:
        x = x.detach().cpu().numpy()
    except:
        pass

    if sp.issparse(x):
        x = x.todense()
        x = x / (np.sqrt(np.square(x).sum(1))).reshape(-1,1)
        x = sp.csr_matrix(x)
    else:
        x = x / (np.sqrt(np.square(x).sum(1))+1e-10).reshape(-1,1)
    # x = x / x.sum(1).reshape(-1,1)
    try:
        dis = euclidean_distances(x)
        return 0.5 * (dis.sum(1)/(dis.shape[1]-1)).mean()
    except:
        return -1
