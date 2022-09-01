#!/usr/bin/env python3
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch as th
import numpy as np
import timeit
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils import data as torch_data
from hype.graph import eval_reconstruction

from hype.lorentz import LorentzManifold
from hype.lorentz_product import LorentzProductManifold
from hype.group_rie import GroupRieManifold
from hype.group_rie_high import GroupRiehighManifold
from hype.bugaenko6 import Bugaenko6Manifold
#from hype.vinberg17 import Vinberg17Manifold #depreciated
#from hype.vinberg3 import Vinberg3Manifold #depreciated
from hype.hyperbolicspace import HyperbolicSpace, normalize_hyperbolicspace
from hype.group_euc import GroupEucManifold
from hype.halfspace_rie import HalfspaceRieManifold
from hype.euclidean import EuclideanManifold
from hype.poincare import PoincareManifold
from hype.discrete import best_nbr

import hype.reflection_sets as reflection_sets

# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

L = th.tensor([1], dtype = th.double)

MANIFOLDS = {
    'lorentz': LorentzManifold,
    'lorentz_product': LorentzProductManifold,
    'group_rie': GroupRieManifold,
    'group_rie_high': GroupRiehighManifold,
    'bugaenko6': Bugaenko6Manifold,
    'vinberg17': HyperbolicSpace,
    'vinberg3' : HyperbolicSpace,
    'hyperbolic_cube' : HyperbolicSpace,
    'group_euc': GroupEucManifold,
    'halfspace_rie': HalfspaceRieManifold,
    'euclidean': EuclideanManifold,
    'poincare': PoincareManifold
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

def move_to_hyperboloid(g,a=1):
	g[0] = th.sqrt((1/a)*(1+(g[1:]*g[1:]).sum()))
	return g

def normalize_bugaenko6(g,g_matrix):

	R, norms, r = reflection_sets.bugaenko6()
	# R is of type (2,34,7,7)
	# norms is of type (2,34)
	a = 1+np.sqrt(2)

	#a point inside a polyhedron (in fact a vertex actually - to do : find a point inside)
	#used to debug to check if distance d(x0,RVIg) decreses 
	#x0 = th.zeros(7, dtype=th.double)
	#for i in range(7): x0[i] = 0
	#x0[0] = 1/np.sqrt(a)
	#x0 = move_to_hyperboloid(x0,a)

	g = move_to_hyperboloid(g,a)
	RVIg =  g.clone()

	normsi = th.zeros_like(norms)
	normsi[0] = norms[0]/(norms[0]*norms[0]-2*norms[1]*norms[1])
	normsi[1] = -norms[1]/(norms[0]*norms[0]-2*norms[1]*norms[1])

	N = 34

	for i in range(N): 
		x = normsi[0,i]*R[0,i] + 2*normsi[1,i]*R[1,i]
		y = normsi[1,i]*R[0,i] + normsi[0,i]*R[1,i]
		R[0,i] = x
		R[1,i] = y
	
	RV = th.zeros_like(R[:,0,:,:])
	RVI = th.zeros_like(R[:,0,:,:])
	dim = 7
	for i in range(dim): 
		RV[0,i,i] = 1
		RVI[0,i,i] = 1
	
	shorter = True
	j = 0

	r_float = th.zeros_like(r[0,:,:])

	for i in range(N):
		r_float[i] = r[0,i] + r[1,i]*np.sqrt(2)

	#to debug: check is the distance of RVIg decreases
	#print(np.arccosh(-LorentzManifold.ldot(x0.double(),RVIg,a)), g)
	
	while shorter:
		shorter = False
		i = 0
		j = j+1

		lprod = LorentzManifold.ldot(RVIg,r_float[i],a)

		while 0 >= lprod:
			i = i+1
			if i == N: break
			lprod = LorentzManifold.ldot(RVIg,r_float[i],a)
		else:
			RV = matmul_constructible(RV,R[:,i])
			RVI = matmul_constructible(R[:,i],RVI)
			#Norm = matmul_constructible(Norm,norms[:,i,:])
			shorter = True

		RVI_float = RVI[0] + RVI[1]*np.sqrt(2)
		RVIg = th.matmul(RVI_float,g)

		#RVIg = move_to_hyperboloid(RVIg,a)
		#print('RV', RV)
		#print('NSorm', Norm)
		#if i<34: print(R[:,i], j)
		#check distance of RVIg to x_0
		#print(np.arccosh(-LorentzManifold.ldot(x0.double(),move_to_hyperboloid(RVIg,a),a)).item(), RVIg.max().item()\
		#, RVI[0].max().item(), RVI[1].max().item(), i, j)
		#print(RVIg, j)

	#NS = (Norm[0,0] + np.sqrt(2)*Norm[1,0]).squeeze(-1)
	#RVIg = RVIg/NS

	RVIg = move_to_hyperboloid(RVIg,a)

	#print(Norm)

	return RVIg, matmul_constructible(g_matrix, RV)

def matmul_constructible(L,R):
	'''Takes two pair of matrices L[0] and L[1], R[0] and R[1]
	representing L = L[0]+sqrt(2)*L[1], same for R
	returns L*R writen as two integer matrices LR[0] and LR[1]'''
	LR = th.zeros_like(L)
	LR[0] = th.matmul(L[0],R[0])+2*th.matmul(L[1],R[1])
	LR[1] = th.matmul(L[1],R[0])+th.matmul(L[0],R[1])
	
	return LR

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

def normalize_bugaenko6_gmatrix(gu, gu_int_matrix):
	uu = th.zeros_like(gu)
	uu_int_matrix = th.zeros_like(gu_int_matrix)
	#uu_int_norm = th.zeros_like(gu_int_norm)
	for i in range(gu.size(0)):
		uu[i], uu_int_matrix[i] = normalize_bugaenko6(gu[i], gu_int_matrix[i])
	return uu, uu_int_matrix

'''
def normalize_vinberg17_gmatrix(gu, gu_int_matrix):
	uu = th.zeros_like(gu)
	uu_int_matrix = th.zeros_like(gu_int_matrix)
	#uu_int_norm = th.zeros_like(gu_int_norm)
	for i in range(gu.size(0)):
		uu[i], uu_int_matrix[i] = normalize_vinberg17(gu[i], gu_int_matrix[i])
	return uu, uu_int_matrix

def normalize_vinberg3_gmatrix(gu, gu_int_matrix):
    uu = th.zeros_like(gu)
    uu_int_matrix = th.zeros_like(gu_int_matrix)
    #uu_int_norm = th.zeros_like(gu_int_norm)
    for i in range(gu.size(0)):
        uu[i], uu_int_matrix[i] = normalize_vinberg3(gu[i], gu_int_matrix[i])
        #if i % 10 == 0: print(i)
    return uu, uu_int_matrix
'''

def normalize_hyperbolicspace_gmatrix(gu, gu_int_matrix,polytope,dim):
    uu = th.zeros_like(gu)
    uu_int_matrix = th.zeros_like(gu_int_matrix)
    #uu_int_norm = th.zeros_like(gu_int_norm)
    for i in range(gu.size(0)):
        uu[i], uu_int_matrix[i] = normalize_hyperbolicspace(gu[i], gu_int_matrix[i],polytope,dim)
        #if i % 10 == 0: print(i)
    #print(gu_int_matrix[0])
    return uu, uu_int_matrix

'''
def normalize_halfspace(g):
    y = th.zeros(g.size())
    n = (g.size(0)-1)//2
    a = th.floor(th.log2(g[n-1]))
    y[-1] = g[-1] + a
    y[n:-2] = th.floor(2**(-1*a) * (g[:n-1] + g[n:-2]))
    y[:n] = 2**(-1*a) * (g[:n]+g[n:-1]) - y[n:-1]
    assert y[-2]==0
    return y
'''

def normalize_halfspace_matrix(g):
    y = th.zeros(g.size())
    d = (g.size(-1)-1)//2
    a = th.floor(th.log2(g[...,d-1]))#n
    y[...,-1] = g[...,-1] + a#n
    y[...,d:-2] = th.floor(2**(-1*a).unsqueeze(-1).expand_as(g[...,:d-1]) * (g[...,:d-1] + g[...,d:-2]))#n*(d-1)
    y[...,:d] = 2**(-1*a).unsqueeze(-1).expand_as(g[...,:d]) * g[...,d:-1] - y[...,d:-1] + 2**(-1*a).unsqueeze(-1).expand_as(g[...,:d]) * g[...,:d]#n*d
    assert y[...,-2].max()==0
    return y

def normalize(epoch, model, opt):
    if opt.nor=='group':
        NMD, NMD_int_matrix = normalize_gmatrix(model.lt.weight.data.cpu().clone(), model.int_matrix.data.cpu().clone())
        model.int_matrix.data.copy_(NMD_int_matrix)
        model.lt.weight.data.copy_(NMD)
    elif opt.nor == 'HyperbolicSpace':
        NMD, NMD_int_matrix = normalize_hyperbolicspace_gmatrix(model.lt.weight.data.cpu().clone(), model.int_matrix.data.cpu().clone(),opt.polytope,opt.dim)
        model.int_matrix.data.copy_(NMD_int_matrix)
        model.lt.weight.data.copy_(NMD)
    elif opt.nor == 'bugaenko6':
        NMD, NMD_int_matrix = normalize_bugaenko6_gmatrix(model.lt.weight.data.cpu().clone(), model.int_matrix.data.cpu().clone())
        model.int_matrix.data.copy_(NMD_int_matrix)
        model.lt.weight.data.copy_(NMD)
    elif opt.nor == 'halfspace':
        NMD = normalize_halfspace_matrix(model.lt.weight.data.clone())
        model.lt.weight.data.copy_(NMD)

def train(
        device,
        model,
        data,
        optimizer,
        opt,
        log,
        model_comp=False,
        optimizer_comp=False,
        opt_comp=False,
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

    if model_comp:
        epoch_loss_comp = th.Tensor(len(loader))
        #counts = th.zeros(model.nobjects, 1).to(device)

        LOSS_comp = np.zeros(opt.epochs)

    for epoch in range(opt.epoch_start, opt.epochs):

        max_grad = 0
        if epoch % 10 == 0:   
            log.info(f'max coordinate: {th.abs(model.lt.weight.data).max().item()}')
            log.info(f'fifth vertex: {model.lt.weight.data[5]}')

        if 'bugaenko6' in opt.polytope or 'group' in opt.polytope or 'vinberg17' in opt.polytope or 'vinberg3' in opt.polytope:
            log.info(f'max and min matrix coef: {model.int_matrix.max().item()}, {model.int_matrix.min().item()}')
        
        epoch_loss.fill_(0)
        data.burnin = False
        t_start = timeit.default_timer()
        lr = opt.lr 
        if epoch < opt.burnin:
            data.burnin = True
            lr = opt.lr * _lr_multiplier

        #for every node i compute its new representation (v_i,M) where v_i is in the polytope and M is a matrix.
        if opt.optimalisation == 'discrete':

            #if epoch < opt.start_discrete we use rsgd and a tiling model (x,M).
            if (epoch>opt.stre and (epoch-opt.stre)%opt.norevery==0 and epoch < opt.start_discrete) or epoch==0:
                normalize(epoch, model, opt)

            #for each vertex v, we copy to v the first colum of the matrix M_v
            if epoch == opt.start_discrete:
                normalize(epoch, model, opt)
                model.lt.weight.data.copy_(model.int_matrix.narrow(2,0,1).squeeze())
                
        elif (opt.nor!='none' and epoch>opt.stre and (epoch-opt.stre)%opt.norevery==0) or epoch==0:
            normalize(epoch, model, opt)
            #d = model.int_matrix[0].size(0)
            #G = th.eye(d)
            #G[0,0] = -1
            #print(model.int_matrix[0])
            #print(model.int_matrix[0]@G@model.int_matrix[0].t())
            #print(model.int_matrix[10])
            #print(model.int_matrix[10]@G@model.int_matrix[10].t())

        #print(model.lt.weight.data[0])
        #print(model.lt.weight.data[10])
        #print(LorentzManifold.ldot(model.lt.weight.data[10],model.lt.weight.data[10]))
        #print(LorentzManifold.ldot(model.lt.weight.data[0],model.lt.weight.data[0]))

        loader_iter = tqdm(loader) if progress else loader
        for i_batch, (inputs, targets) in enumerate(loader_iter):
            elapsed = timeit.default_timer() - t_start

            inputs = inputs.to(device)
            targets = targets.to(device)

            # count occurrences of objects in batch, ignore if model_comp != False
            if hasattr(opt, 'asgd') and opt.asgd: 
                counts = th.bincount(inputs.view(-1), minlength=model.nobjects)
                counts.clamp_(min=1).div_(inputs.size(0))
                counts = counts.double().unsqueeze(-1)

            if model_comp:
                optimizer_comp.zero_grad()
                preds_comp = model_comp(inputs)

            if 'discrete' in opt.optimalisation and epoch >= opt.start_discrete:
                vertices = inputs[:,0]
                model.lt.weight.data[vertices], loss = best_nbr(model.lt.weight.data[inputs],opt.polytope,opt.dim)
                
            else:   
                optimizer.zero_grad()
                preds = model(inputs)
                loss = model.loss(preds, targets, size_average=True)
                loss.backward()            
                if max_grad < th.abs(model.lt.weight.grad.to_dense().max()): max_grad = th.abs(model.lt.weight.grad.to_dense().max())
                optimizer.step(lr, counts=counts)

            epoch_loss[i_batch] = loss.cpu().item()

            if model_comp:
                loss_comp = model_comp.loss(preds_comp, targets, size_average=True)
                loss_comp.backward()
                optimizer_comp.step(lr=lr, counts=counts)
                epoch_loss_comp[i_batch] = loss_comp.cpu().item()
        
        
        LOSS[epoch] = th.mean(epoch_loss).item()

        if model_comp:
            LOSS_comp[epoch] = th.mean(epoch_loss_comp).item()

        log.info('json_stats: {'
                 f'"epoch": {epoch}, '
                 f'"elapsed": {elapsed}, '
                 f'"loss": {LOSS[epoch]}, '
                 f'"max grad": {max_grad}'
                 '}')

        if model_comp:
            log.info('json_stats: {'
                 f'"loss_comp": {LOSS_comp[epoch]}, '
                 '}')

        if (epoch+1)%opt.eval_each==0:
            manifold = MANIFOLDS[opt.manifold](debug=opt.debug, max_norm=opt.maxnorm)
            if 'group' in opt.manifold:
                meanrank, maprank = eval_reconstruction(opt.adj, model.lt.weight.data.clone(), manifold.distance, lt_int_matrix = model.int_matrix.data.clone())
                sqnorms = manifold.pnorm(model.lt.weight.data.clone(), model.int_matrix.data.clone())
            elif 'HyperbolicSpace' in opt.nor or 'bugaenko6' in opt.nor:
                meanrank, maprank = eval_reconstruction(opt.adj, model.lt.weight.data.clone(), manifold.distance, g = model.g, lt_int_matrix = model.int_matrix.data.clone())
                sqnorms = manifold.pnorm(model.lt.weight.data.clone(), model.int_matrix.data.clone())
            	
                if opt.evaluate_int_coordinates:
                    int_coordinates = model.int_matrix.data.clone().narrow(-1,0,1).squeeze()
                    int_meanrank, int_maprank = eval_reconstruction(opt.adj, int_coordinates, LorentzManifold().distance)

                #imax = th.argmax(sqnorms).item()
            	#imin = th.argmin(sqnorms).item()
            else:
                meanrank, maprank = eval_reconstruction(opt.adj, model.lt.weight.data.clone(), manifold.distance)
                sqnorms = manifold.pnorm(model.lt.weight.data.clone())

            if 'group' in opt.manifold:    
                print('max matrix:\n{}\n,min matrix:\n{}'.format(model.int_matrix[imax],model.int_matrix[imin]))
                print('max: {}, min: {}'.format(model.int_matrix.max().item(),model.int_matrix.min().item()))

            log.info(
                'json_stats: {'         
                f'"sqnorm_min": {sqnorms.min().item()}, '
                f'"sqnorm_avg": {sqnorms.mean().item()}, '
                f'"sqnorm_max": {sqnorms.max().item()}, '
                f'"mean_rank": {meanrank}, '
                f'"map_rank": {maprank}, '
                '}'
            )

            if opt.evaluate_int_coordinates:
                log.info(
                    f'"int_mean_rank": {int_meanrank}, '
                    f'"int_map_rank": {int_maprank}, '
                )

            #make_plot(epoch,model.lt.weight.data,opt)
            

    #print('LOSS \n', LOSS)
    
    if model_comp: 
        print('LOSS_comp \n', LOSS_comp)

def make_plot(epoch,model_data,opt):
    if (epoch+1)%500 == 0: 
            data_in_ball = LorentzManifold().to_poincare_ball(model_data)
            X = data_in_ball.narrow(-1,0,1).squeeze().numpy()
            Y = data_in_ball.narrow(-1,1,1).squeeze().numpy()
            plt.scatter(X,Y)
            for v in opt.adj: 
                for w in opt.adj[v]: 
                    plt.plot([X[v],X[w]], [Y[v],Y[w]],color='blue')
            plt.show()
