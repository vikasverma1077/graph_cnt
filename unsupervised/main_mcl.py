# Optional: eliminating warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from arguments import arg_parse
from cortex_DIM.nn_modules.mi_networks import MIFCNet, MI1x1ConvNet
from evaluate_embedding import evaluate_embedding
from gin import Encoder
from losses import local_global_loss_
from model import *# FF, PriorDiscriminator, Projhead, mixup_data, 
from torch import optim
from torch.autograd import Variable
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
import json
import json
import numpy as np
import os.path as osp
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from nt_xent import NTXentLoss

class MCL(nn.Module):
  def __init__(self, hidden_dim, num_gc_layers, args, alpha=0.5, beta=1., gamma=.1):
    super(MCL, self).__init__()

    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    self.prior = args.prior

    self.embedding_dim = mi_units = hidden_dim * num_gc_layers
    self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)
    
    self.proj_head = Projhead(self.embedding_dim)
    self.nt_xent_loss  = NTXentLoss( batch_size= args.batch_size, temperature=args.temperature, use_cosine_similarity=True)

    self.local_d = FF(self.embedding_dim)
    self.global_d = FF(self.embedding_dim)
    
    # self.local_d = MI1x1ConvNet(self.embedding_dim, mi_units)
    # self.global_d = MIFCNet(self.embedding_dim, mi_units)

    if self.prior:
        self.prior_d = PriorDiscriminator(self.embedding_dim)

    self.init_emb()

  def init_emb(self):
    initrange = -1.5 / self.embedding_dim
    for m in self.modules():
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

  def forward(self, x, edge_index, batch, num_graphs):
    # batch_size = data.num_graphs
    if x is None:
        x = torch.ones(batch.shape[0]).to(device)

    y, M = self.encoder(x, edge_index, batch) # y is global rep, M is node level rep
    """
    ### get mixed samples ###
    y_positive1 = mixup_data(y, args.mixup_alpha)
    y_positive2 = mixup_data(y, args.mixup_alpha)
    y_mix = torch.cat((y_positive1,y_positive2), 0)
    """
    y_positive1_1 = mixup_data(y, args.mixup_alpha)
    y_positive2_1 = mixup_data(y, args.mixup_alpha)
    y_positive1_2 = mixup_geo_data(y, args.mixup_alpha)
    y_positive2_2 = mixup_geo_data(y, args.mixup_alpha)
    y_positive1_3 = mixup_binary_data(y, args.mixup_alpha)
    y_positive2_3 = mixup_binary_data(y, args.mixup_alpha)
    
    y_positive1 = torch.cat((y_positive1_2,y_positive1_3),0)#,y_positive1_3),0)
    y_positive2 = torch.cat((y_positive2_2,y_positive2_3),0)#,y_positive2_3),0)
    
    size = y_positive1.size()[0]
    index = torch.randperm(size).cuda()
    index = index[0:y.size()[0]]
    y_positive1 = y_positive1[index,:]
    y_positive2 = y_positive2[index,:]
    y_mix = torch.cat((y_positive1,y_positive2), 0)
    
    ## pass the mixed positive features to proj head ##
    hiddens = self.proj_head(y_mix)
    ## get the Simclr loss ##
    hiddens = F.normalize(hiddens, dim=1)
    hiddens1, hiddens2 = torch.split(hiddens,int(hiddens.shape[0]/2),0)
    #print(hiddens1.shape, hiddens2.shape)
    nt_xent_loss = self.nt_xent_loss(hiddens1, hiddens2)
    
    ### get in infograph loss ##
    g_enc = self.global_d(y)
    l_enc = self.local_d(M)

    mode='fd'
    measure='JSD'
    local_global_loss = local_global_loss_(l_enc, g_enc, edge_index, batch, measure)
 
    if self.prior:
        prior = torch.rand_like(y)
        term_a = torch.log(self.prior_d(prior)).mean()
        term_b = torch.log(1.0 - self.prior_d(y)).mean()
        PRIOR = - (term_a + term_b) * self.gamma
    else:
        PRIOR = 0
        
    info_graph_loss = local_global_loss + PRIOR
    
    return  nt_xent_loss
    #return info_graph_loss
if __name__ == '__main__':
    args = arg_parse()
    accuracies = {'logreg':[], 'svc':[], 'linearsvc':[], 'randomforest':[]}
    epochs = args.epochs
    log_interval = 1
    batch_size = args.batch_size
    lr = args.lr
    DS = args.DS
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', DS)
    # kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)

    dataset = TUDataset(path, name=DS).shuffle()
    dataset_num_features = max(dataset.num_features, 1)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    print('================')
    print('lr: {}'.format(lr))
    print('num_features: {}'.format(dataset_num_features))
    print('hidden_dim: {}'.format(args.hidden_dim))
    print('num_gc_layers: {}'.format(args.num_gc_layers))
    print('================')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MCL(args.hidden_dim, args.num_gc_layers, args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.eval()
    emb, y = model.encoder.get_embeddings(dataloader)
    print('===== Before training =====')
    
    res = evaluate_embedding(emb, y)
    accuracies['logreg'].append(res)#(res[0])
    #accuracies['svc'].append(res[1])
    #accuracies['linearsvc'].append(res[2])
    #accuracies['randomforest'].append(res[3])
    
    for epoch in range(1, epochs+1):
        loss_all = 0
        model.train()
        for data in dataloader:
            data = data.to(device)
            optimizer.zero_grad()
            #import pdb; pdb.set_trace()
            loss = model(data.x, data.edge_index, data.batch, data.num_graphs)
            loss_all += loss.item() * data.num_graphs
            loss.backward()
            optimizer.step()
        print('===== Epoch {}, Loss {} ====='.format(epoch, loss_all / len(dataloader)))

        if epoch % log_interval == 0:
            model.eval()
            emb, y = model.encoder.get_embeddings(dataloader)
            res = evaluate_embedding(emb, y)
            accuracies['logreg'].append(res)#(res[0])
            #accuracies['svc'].append(res[1])
            #accuracies['linearsvc'].append(res[2])
            #accuracies['randomforest'].append(res[3])
            print(accuracies)   
    last_epoch_acc = np.asarray(accuracies['logreg'])[-5:].mean()
    print ('last_epoch_acc=', last_epoch_acc) 
    with open('unsupervised.log', 'a+') as f:
        s = json.dumps(accuracies)
        f.write('{},{},{},{},{},{}\n'.format(args.DS, args.num_gc_layers, epochs, log_interval, lr, s))
