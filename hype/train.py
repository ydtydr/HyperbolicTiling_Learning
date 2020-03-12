#!/usr/bin/env python3
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch as th
import numpy as np
import timeit
from tqdm import tqdm
from torch.utils import data as torch_data
from hype.graph import eval_reconstruction

from hype.Euclidean import EuclideanManifold
from hype.Poincare import PoincareManifold
from hype.Lorentz import LorentzManifold
from hype.Halfspace import HalfspaceManifold
from hype.NLorentz import NLorentzManifold
from hype.LTiling_rsgd import LTilingRSGDManifold
from hype.NLTiling_rsgd import NLTilingRSGDManifold
from hype.LTiling_sgd import LTilingSGDManifold
from hype.HTiling_rsgd import HTilingRSGDManifold
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

MANIFOLDS = {
    'Euclidean': EuclideanManifold,
    'Poincare': PoincareManifold,
    'Lorentz': LorentzManifold,
    'Halfspace': HalfspaceManifold,
    'NLorentz': NLorentzManifold,
    'LTiling_rsgd': LTilingRSGDManifold,
    'NLTiling_rsgd': NLTilingRSGDManifold,
    'LTiling_sgd': LTilingSGDManifold,
    'HTiling_rsgd': HTilingRSGDManifold
}


_lr_multiplier = 0.01

def normalize_g(g, g_int_matrix):
    L = th.sqrt(th.Tensor([[3.0,0,0],[0,1.0,0],[0,0,1.0]]))
    R = th.sqrt(th.Tensor([[1.0/3.0,0,0],[0,1.0,0],[0,0,1.0]]))
    ga = th.LongTensor([[2,1,0],[0,0,-1],[3,2,0]])
    gb = th.LongTensor([[2,-1,0],[0,0,-1],[-3,2,0]])
    gai = th.LongTensor([[2,0,-1],[-3,0,2],[0,-1,0]])
    gbi = th.LongTensor([[2,0,1],[3,0,2],[0,-1,0]])
    RVI = th.LongTensor([[1,0,0],[0,1,0],[0,0,1]])
    RV = th.LongTensor([[1,0,0],[0,1,0],[0,0,1]])
    gmat = g_int_matrix
    x = g[:3].clone()
    x[0]=th.sqrt(1+x[1]**2+x[2]**2)
    y=x.clone()
    while ((2 * x[1] ** 2 - x[2] ** 2 - 1 > 0) or (2 * x[2] ** 2 - x[1] ** 2 - 1 > 0)):
        prex = x.clone()
        preRV = RV.clone()
        if x[1] <= -th.abs(x[2]):
            RVI = th.matmul(ga, RVI)
            RV = th.matmul(RV, gai)
        elif x[1] >= th.abs(x[2]):
            RVI = th.matmul(gb, RVI)
            RV = th.matmul(RV, gbi)
        elif x[2] < -th.abs(x[1]):
            RVI = th.matmul(gbi, RVI)
            RV = th.matmul(RV, gb)
        elif x[2] > th.abs(x[1]):
            RVI = th.matmul(gai, RVI)
            RV = th.matmul(RV, ga)
#         x = th.matmul(L, th.matmul(RVI.float(), th.matmul(R, y.unsqueeze(-1)))).squeeze(-1)
        if L.dtype == th.float64:
            x = th.matmul(L, th.matmul(RVI.double(), th.matmul(R, y.unsqueeze(-1)))).squeeze(-1)
        elif L.dtype == th.float32:
            x = th.matmul(L, th.matmul(RVI.float(), th.matmul(R, y.unsqueeze(-1)))).squeeze(-1)
        x[0] = th.sqrt(1 + x[1] ** 2 + x[2] ** 2)
        if x[0] > prex[0]:
            if L.dtype == th.float64:
                return prex, th.matmul(gmat, preRV.double())
            elif L.dtype == th.float32:
                return prex, th.matmul(gmat, preRV.float())
    if L.dtype == th.float64:
        return x, th.matmul(gmat, RV.double())
    elif L.dtype == th.float32:
        return x, th.matmul(gmat, RV.float())


def normalize_gmatrix(gu, gu_int_matrix):
    uu = th.zeros_like(gu)
    uu_int_matrix = th.zeros_like(gu_int_matrix)
    if len(gu_int_matrix.size())==4:
        for i in range(gu.size(0)):
            for j in range(uu_int_matrix.size(1)):
                uu[i,3*j:3*(j+1)], uu_int_matrix[i,j] = normalize_g(gu[i,3*j:3*(j+1)], gu_int_matrix[i,j])
    else:
        for i in range(gu.size(0)):
            uu[i], uu_int_matrix[i] = normalize_g(gu[i], gu_int_matrix[i])
    return uu, uu_int_matrix

def normalize_halfspace(g):
    y = th.zeros(g.size())
    n = (g.size(0)-1)//2
    a = th.floor(th.log2(g[n-1]))
    y[-1] = g[-1] + a
    y[n:-2] = th.floor(2**(-1*a) * (g[:n-1] + g[n:-2]))
    y[:n] = 2**(-1*a) * (g[:n]+g[n:-1]) - y[n:-1]
    assert y[-2]==0
    return y

def normalize_halfspace_matrix(g):
    y = th.zeros(g.size())
    d = (g.size(-1)-1)//2
    a = th.floor(th.log2(g[...,d-1]))#n
    y[...,-1] = g[...,-1] + a#n
    y[...,d:-2] = th.floor(2**(-1*a).unsqueeze(-1).expand_as(g[...,:d-1]) * (g[...,:d-1] + g[...,d:-2]))#n*(d-1)
    y[...,:d] = 2**(-1*a).unsqueeze(-1).expand_as(g[...,:d]) * g[...,d:-1] - y[...,d:-1] + 2**(-1*a).unsqueeze(-1).expand_as(g[...,:d]) * g[...,:d]#n*d
    assert y[...,-2].max()==0
    return y

def train(
        thread_id,
        device,
        model,
        data,
        optimizer,
        opt,
        log,
        progress=False
):
    if isinstance(data, torch_data.Dataset):
        loader = torch_data.DataLoader(data, batch_size=opt.batchsize,
            shuffle=False, num_workers=opt.ndproc)
    else:
        loader = data

    epoch_loss = th.Tensor(len(loader))
    counts = th.zeros(model.nobjects, 1).to(device)

    LOSS = np.zeros(opt.epochs)        
    for epoch in range(opt.epoch_start, opt.epochs):
        print(th.abs(model.lt.weight.data).max().item())
        
        epoch_loss.fill_(0)
        data.burnin = False
        t_start = timeit.default_timer()
        lr = opt.lr        
        if epoch < opt.burnin:
            data.burnin = True
            lr = opt.lr * _lr_multiplier
            
            
        loader_iter = tqdm(loader) if progress else loader
        for i_batch, (inputs, targets) in enumerate(loader_iter):
            elapsed = timeit.default_timer() - t_start

            inputs = inputs.to(device)
            targets = targets.to(device)

            # count occurrences of objects in batch
            if hasattr(opt, 'asgd') and opt.asgd:
                counts = th.bincount(inputs.view(-1), minlength=model.nobjects)
                counts.clamp_(min=1).div_(inputs.size(0))
                counts = counts.double().unsqueeze(-1)

            optimizer.zero_grad()
            preds = model(inputs)

            loss = model.loss(preds, targets, size_average=True)
            loss.backward()
            optimizer.step(lr=lr, counts=counts)
            epoch_loss[i_batch] = loss.cpu().item()
        LOSS[epoch] = th.mean(epoch_loss).item()
        log.info('json_stats: {'
                 f'"thread_id": {thread_id}, '
                 f'"epoch": {epoch}, '
                 f'"elapsed": {elapsed}, '
                 f'"loss": {LOSS[epoch]}, '
                 '}')
        if opt.nor!='none' and epoch>opt.stre and (epoch-opt.stre)%opt.norevery==0:
            if opt.nor=='LTiling':
                NMD, NMD_int_matrix = normalize_gmatrix(model.lt.weight.data.cpu().clone(), model.int_matrix.data.clone())
                model.int_matrix.data.copy_(NMD_int_matrix)
                model.lt.weight.data.copy_(NMD)
            elif opt.nor == 'HTiling':
                NMD = normalize_halfspace_matrix(model.lt.weight.data.clone())
                model.lt.weight.data.copy_(NMD)
        
#         if (epoch+1)%opt.eval_each==0 and thread_id==0:
#             manifold = MANIFOLDS[opt.manifold](debug=opt.debug, max_norm=opt.maxnorm)
#             if 'LTiling' in opt.manifold:
#                 meanrank, maprank = eval_reconstruction(opt.adj, model.lt.weight.data.clone(), manifold.distance, lt_int_matrix = model.int_matrix.data.clone(), workers = opt.ndproc)
#                 sqnorms = manifold.pnorm(model.lt.weight.data.clone(), model.int_matrix.data.clone())
#             else:
#                 meanrank, maprank = eval_reconstruction(opt.adj, model.lt.weight.data.clone(), manifold.distance)#, workers = opt.ndproc)
#                 sqnorms = manifold.pnorm(model.lt.weight.data.clone())
#             log.info(
#                 'json_stats during training: {'
#                 f'"sqnorm_min": {sqnorms.min().item()}, '
#                 f'"sqnorm_avg": {sqnorms.mean().item()}, '
#                 f'"sqnorm_max": {sqnorms.max().item()}, '
#                 f'"mean_rank": {meanrank}, '
#                 f'"map": {maprank}, '
#                 '}'
#             )

            
#     print(LOSS)
