import torch
import torch.nn as nn
import torch.nn.functional as F
from cortex_DIM.functions.gan_losses import get_positive_expectation, get_negative_expectation

LARGE_NUM = 1e9

def local_global_loss_(l_enc, g_enc, edge_index, batch, measure):
    '''
    Args:
        l: Local feature map.
        g: Global features.
        measure: Type of f-divergence. For use with mode `fd`
        mode: Loss mode. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.
    Returns:
        torch.Tensor: Loss.
    '''
    num_graphs = g_enc.shape[0]
    num_nodes = l_enc.shape[0]

    pos_mask = torch.zeros((num_nodes, num_graphs)).cuda()
    neg_mask = torch.ones((num_nodes, num_graphs)).cuda()
    for nodeidx, graphidx in enumerate(batch):
        pos_mask[nodeidx][graphidx] = 1.
        neg_mask[nodeidx][graphidx] = 0.

    res = torch.mm(l_enc, g_enc.t())

    E_pos = get_positive_expectation(res * pos_mask, measure, average=False).sum()
    E_pos = E_pos / num_nodes
    E_neg = get_negative_expectation(res * neg_mask, measure, average=False).sum()
    E_neg = E_neg / (num_nodes * (num_graphs - 1))

    return E_neg - E_pos

def adj_loss_(l_enc, g_enc, edge_index, batch):
    num_graphs = g_enc.shape[0]
    num_nodes = l_enc.shape[0]

    adj = torch.zeros((num_nodes, num_nodes)).cuda()
    mask = torch.eye(num_nodes).cuda()
    for node1, node2 in zip(edge_index[0], edge_index[1]):
        adj[node1.item()][node2.item()] = 1.
        adj[node2.item()][node1.item()] = 1.

    res = torch.sigmoid((torch.mm(l_enc, l_enc.t())))
    res = (1-mask) * res
    # print(res.shape, adj.shape)
    # input()

    loss = nn.BCELoss()(res, adj)
    return loss

def contrastive_loss(hiddens,
                     hidden_norms=True,
                     temperature=1.0
                     ):
    if hidden_norms:
        hiddens =  torch.norm(hiddens, dim=1)
    hiddens1, hiddens2 = torch.split(hiddens,2,0)
    batch_size = hiddens1.shape[0]
    labels_idx = torch.range(0, batch_size-1, dtype=torch.long)
    labels = to_one_hot(labels_idx, batch_size*2)
    masks = to_one_hot(labels_idx, batch_size)
    
    logits_aa = torch.matmul(hiddens1, torch.transpose(hiddens1,0,1))/temperature
    logits_aa = logits_aa - masks * LARGE_NUM
    
    logits_bb = torch.matmul(hiddens2, torch.transpose(hiddens2,0,1))/temperature
    logits_bb = logits_bb - masks * LARGE_NUM
    
    logits_ab = torch.matmul(hiddens1, torch.transpose(hiddens2,0,1))/temperature
    logits_ba = torch.matmul(hiddens2, torch.transpose(hiddens1,0,1))/temperature
    
    #loss_a = torch.nn.cross
    #unfinished
    
    


def to_one_hot(inp,one_hot_length):
    y_onehot = torch.FloatTensor(inp.size(0), one_hot_length)
    y_onehot.zero_()

    y_onehot.scatter_(1, inp.unsqueeze(1).data.cpu(), 1)
    
    return Variable(y_onehot.cuda(),requires_grad=False)