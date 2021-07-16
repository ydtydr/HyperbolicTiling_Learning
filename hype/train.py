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

from hype.lorentz import LorentzManifold
from hype.lorentz_product import LorentzProductManifold
from hype.group_rie import GroupRieManifold
from hype.group_rie_high import GroupRiehighManifold
from hype.bugaenko6 import Bugaenko6Manifold
from hype.vinberg17 import Vinberg17Manifold
from hype.vinberg3 import Vinberg3Manifold
from hype.group_euc import GroupEucManifold
from hype.halfspace_rie import HalfspaceRieManifold
from hype.euclidean import EuclideanManifold
from hype.poincare import PoincareManifold

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
    'vinberg17': Vinberg17Manifold,
    'vinberg3' : Vinberg3Manifold,
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

def normalize_vinberg(g,g_matrix, R, norms, r, x0, debug=0):
    '''For vinberg17, Only R[18] is not integral, but the fractional part is 0.2 '''

    g = move_to_hyperboloid(g)
    RVIg =  g.clone()

    number_of_roots = R.size(0)
    R = th.div(R,norms.unsqueeze(-1).unsqueeze(-1))

    RV = th.zeros_like(R[0,:,:])
    RVI = th.zeros_like(R[0,:,:])
    dim = R.size(-1)

    for i in range(dim): 
        RV[i,i] = 1
        RVI[i,i] = 1
	
    #j = 0
    #print(np.arccosh(-LorentzManifold.ldot(x0.double(),RVIg,a)), g)
    
    while True:
        
        #distance of g to the hyperplane orthogonal to the root r = arccosh(sqrt(1+<g,r>^2/<r,r>^2))
        #we want to apply a reflection R on g iff the distance to the hyperplane is the biggest,
        #at least arccosh(2) and g is on the other side of the hyperplane then the polytope. 
        max_index = -1
        max_lprod = -1

        for i in range(number_of_roots):
            lprod = LorentzManifold.ldot(RVIg,r[i])
            if lprod >= norms[i] and lprod > max_lprod: 
                max_lprod = lprod
                max_index = i
        
        if max_index == -1: break
        else:
            RV = th.matmul(RV,R[max_index])
            RVI = th.matmul(R[max_index],RVI)
            RVIg = th.matmul(RVI,g)
            RVIg = move_to_hyperboloid(RVIg)
            #Norm = matmul_constructible(Norm,norms[:,i,:])

        # i = 0
        # j = j+1

        # shorter = False
        # lprod = LorentzManifold.ldot(RVIg,r[i])

        # while norms[i] >= lprod:  
        #     i = i+1
        #     if i == N: break
        #     lprod = LorentzManifold.ldot(RVIg,r[i])
        # else:
        #     RV = th.matmul(RV,R[i])
        #     RVI = th.matmul(R[i],RVI)
        #     #Norm = matmul_constructible(Norm,norms[:,i,:])
        #     shorter = True

		#RVIg = move_to_hyperboloid(RVIg,a)
		#print('RV', RV)
		#print('NSorm', Norm)
		#if i<34: print(R[:,i], j)
		#check distance of RVIg to x_0
        if debug==1: print(np.arccosh(-LorentzManifold.ldot(x0,RVIg)).item(), RVIg.max().item(), max_index)
		#, RVI[0].max().item(), RVI[1].max().item(), i, j)
		#print(RVIg, j)

    if debug==1:	
        print('RV', RV)

        for i in range(number_of_roots):
            print('<r,g>', LorentzManifold.ldot(r[i],RVIg).item() < norms[i])

    return RVIg, th.matmul(g_matrix, RV)

def normalize_vinberg17(g,g_matrix,debug=0):

    #a point used to debug to check if distance d(x0,RVIg) decreses
    x0 = th.zeros(18)
    for i in range(18): x0[i] = 0
    x0 = move_to_hyperboloid(x0)

    R, norms, r = reflection_sets.vinberg17()

    return normalize_vinberg(g,g_matrix, R, norms, r, x0, debug)

def normalize_vinberg3(g,g_matrix,debug=0):

    #a point used to debug to check if distance d(x0,RVIg) decreses
    x0 = th.zeros(4).double()
    for i in range(4): x0[i] = 0
    x0 = move_to_hyperboloid(x0)

    R, norms, r = reflection_sets.vinberg3()

    return normalize_vinberg(g,g_matrix, R, norms, r, x0, debug)
    
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
        print(th.abs(model.lt.weight.data).max().item())
        if model_comp:
            print(th.abs(model_comp.lt.weight.data).max().item())
            if 'bugaenko6' in opt_comp.manifold or 'vinberg17' in opt_comp.manifold or 'vinberg3' in opt_comp.manifold:
                print(th.abs(model_comp.int_matrix).max().item())

        #if 'bugaenko6' in opt.manifold or 'group' in opt.manifold or 'vinberg17' in opt.manifold:
        #	print(model.int_matrix.max().item(), model.int_matrix.min().item())

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

            # count occurrences of objects in batch, ignore if model_comp != False
            if hasattr(opt, 'asgd') and opt.asgd: 
                counts = th.bincount(inputs.view(-1), minlength=model.nobjects)
                counts.clamp_(min=1).div_(inputs.size(0))
                counts = counts.double().unsqueeze(-1)

            optimizer.zero_grad()
            preds = model(inputs)

            if model_comp:
                optimizer_comp.zero_grad()
                preds_comp = model_comp(inputs)

            #print(preds)

            loss = model.loss(preds, targets, size_average=True)
            loss.backward()
            if max_grad < th.abs(model.lt.weight.grad.to_dense().max()): max_grad = th.abs(model.lt.weight.grad.to_dense().max())
            optimizer.step(lr=lr, counts=counts)
            epoch_loss[i_batch] = loss.cpu().item()

            if model_comp:
                loss_comp = model_comp.loss(preds_comp, targets, size_average=True)
                loss_comp.backward()
                optimizer_comp.step(lr=lr, counts=counts)
                epoch_loss_comp[i_batch] = loss_comp.cpu().item()
        
        print(max_grad)
        LOSS[epoch] = th.mean(epoch_loss).item()

        if model_comp:
            LOSS_comp[epoch] = th.mean(epoch_loss_comp).item()

        log.info('json_stats: {'
                 f'"epoch": {epoch}, '
                 f'"elapsed": {elapsed}, '
                 f'"loss": {LOSS[epoch]}, '
                 '}')

        if model_comp:
            log.info('json_stats: {'
                 f'"loss_comp": {LOSS_comp[epoch]}, '
                 '}')

        if (opt.nor!='none' and epoch>opt.stre and (epoch-opt.stre)%opt.norevery==0) or epoch==0:
            if opt.nor=='group':
                NMD, NMD_int_matrix = normalize_gmatrix(model.lt.weight.data.cpu().clone(), model.int_matrix.data.cpu().clone())
                model.int_matrix.data.copy_(NMD_int_matrix)
                model.lt.weight.data.copy_(NMD)
            elif opt.nor == 'bugaenko6':
                NMD, NMD_int_matrix = normalize_bugaenko6_gmatrix(model.lt.weight.data.cpu().clone(), model.int_matrix.data.cpu().clone())
               	model.int_matrix.data.copy_(NMD_int_matrix)
               	model.lt.weight.data.copy_(NMD)
            elif opt.nor == 'vinberg17':
            	print('normalizuje!')
            	NMD, NMD_int_matrix = normalize_vinberg17_gmatrix(model.lt.weight.data.cpu().clone(), model.int_matrix.data.cpu().clone())
            	model.int_matrix.data.copy_(NMD_int_matrix)
            	model.lt.weight.data.copy_(NMD)
            elif opt.nor == 'vinberg3':
                print('normalizuje!')
                NMD, NMD_int_matrix = normalize_vinberg3_gmatrix(model.lt.weight.data.cpu().clone(), model.int_matrix.data.cpu().clone())
                model.int_matrix.data.copy_(NMD_int_matrix)
                model.lt.weight.data.copy_(NMD)
            elif opt.nor == 'halfspace':
                NMD = normalize_halfspace_matrix(model.lt.weight.data.clone())
                model.lt.weight.data.copy_(NMD)
        
            #print(model.int_matrix.data)

        if (epoch+1)%opt.eval_each==0:
            manifold = MANIFOLDS[opt.manifold](debug=opt.debug, max_norm=opt.maxnorm)
            if 'group' in opt.manifold:
                meanrank, maprank = eval_reconstruction(opt.adj, model.lt.weight.data.clone(), manifold.distance, lt_int_matrix = model.int_matrix.data.clone())
                sqnorms = manifold.pnorm(model.lt.weight.data.clone(), model.int_matrix.data.clone())
            elif 'bugaenko6' in opt.manifold or 'vinberg17' in opt.manifold or 'vinberg3' in opt.manifold:
            	meanrank, maprank = eval_reconstruction(opt.adj, model.lt.weight.data.clone(), manifold.distance, g = model.g, lt_int_matrix = model.int_matrix.data.clone())
            	sqnorms = manifold.pnorm(model.lt.weight.data.clone(), model.int_matrix.data.clone())
            	imax = th.argmax(sqnorms).item()
            	imin = th.argmin(sqnorms).item()
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

        if model_comp:
            if opt_comp.nor!='none' and epoch>opt_comp.stre and (epoch-opt_comp.stre)%opt_comp.norevery==0:
                if opt_comp.nor == 'bugaenko6':
                    NMD_comp, NMD_comp_int_matrix = normalize_bugaenko6_gmatrix(model_comp.lt.weight.data.cpu().clone(), model_comp.int_matrix.data.cpu().clone())
                    model_comp.int_matrix.data.copy_(NMD_comp_int_matrix)
                    model_comp.lt.weight.data.copy_(NMD_comp)
                elif opt_comp.nor == 'vinberg17':
                    print('normalising')
                    NMD_comp, NMD_comp_int_matrix = normalize_vinberg17_gmatrix(model_comp.lt.weight.data.cpu().clone(), model_comp.int_matrix.data.cpu().clone())
                    model_comp.int_matrix.data.copy_(NMD_comp_int_matrix)
                    model_comp.lt.weight.data.copy_(NMD_comp)
                elif opt_comp.nor == 'vinberg3':
                    print('normalising')
                    NMD_comp, NMD_comp_int_matrix = normalize_vinberg3_gmatrix(model_comp.lt.weight.data.cpu().clone(), model_comp.int_matrix.data.cpu().clone())
                    model_comp.int_matrix.data.copy_(NMD_comp_int_matrix)
                    model_comp.lt.weight.data.copy_(NMD_comp)
                
            if (epoch+1)%opt_comp.eval_each==0:
                if 'bugaenko6' in opt_comp.manifold or 'vinberg17' in opt_comp.manifold or 'vinberg3' in opt_comp.manifold:
                    meanrank_comp, maprank_comp = eval_reconstruction(opt_comp.adj, model_comp.lt.weight.data.clone(), manifold_comp.distance, g = model_comp.g, lt_int_matrix = model_comp.int_matrix.data.clone())
                    sqnorms_comp = manifold.pnorm(model_comp.lt.weight.data.clone(), model_comp.int_matrix.data.clone())
                    imax_comp = th.argmax(sqnorms_comp).item()
                    imin_comp = th.argmin(sqnorms_comp).item()
                else:
                    meanrank_comp, maprank_comp = eval_reconstruction(opt_comp.adj, model_comp.lt.weight.data.clone(), manifold_comp.distance)
                    sqnorms_comp = manifold.pnorm(model_comp.lt.weight.data.clone())

                log.info(
                    'json_stats: {'
                    f'"sqnorm_min_comp": {sqnorms_comp.min().item()}, '
                    f'"sqnorm_avg_comp": {sqnorms_comp.mean().item()}, '
                    f'"sqnorm_max_comp": {sqnorms_comp.max().item()}, '
                    f'"mean_rank_comp": {meanrank_comp}, '
                    f'"map_rank_comp": {maprank_comp}, '
            '}'
    )
    

    print('LOSS \n', LOSS)
    
    if model_comp: 
        print('LOSS_comp \n', LOSS_comp)
