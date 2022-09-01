import torch as th 
from torch import nn
import numpy as np
import hype.reflection_sets as reflection_sets
from hype.hyperbolicspace import HyperbolicSpace
from hype.lorentz import LorentzManifold

def best_nbr(e,polytope,dim):
    '''
    For each vertex (ie. a center of cetrain tile) from batch (ie. e[i:0]) 
    loops over all centers of its nbring tiles and selects the tile with least cross entropy
    it is done separatelly for each vertex but in a way it can be computed on GPU.   

    Args:
        e: (torch.tensor[batch,negs+2,dim]): coordinates of vertices. For each vertex from a batch, we have 1 nbr and negs non-nbrs 
                e[i,0] = coordinates of ith vertex from a abatch, 
                e[i,1] = coordinates of a vertex connected to e[i:0]
                e[i,2:] = coorindates of negs vertices non-nbrs of e[i:0]
        return: (torch.tensor[batch,dim]): new centers of verites from a batch 
    '''

    o = e.narrow(1, 1, e.size(1) - 1) #negs+1 vertices o.size: (batch,1+negs,dim)
    s = e.narrow(1, 0, 1) #source s.size: (batch,1,dim) - this is good for broadcasting in dist(s,o)

    R, norms, r, x0 = getattr(reflection_sets,polytope)(dim-1)
    R = th.div(R,norms.unsqueeze(-1).unsqueeze(-1))

    batch = e.size(0)

    dist = LorentzManifold().distance
    cross_entropy = th.nn.functional.cross_entropy
    target = th.zeros(batch).long()
    d = dist(s,o)
    
    #min_loss = cross_entropy(-d,target) #size = (batch)
    min_loss = th.nn.functional.cross_entropy(-d,target,reduction='none')
    min_centers = s.squeeze() #size = (batch,dim)

    for i in range(R.size(0)):

        new_s = th.matmul(R[i],s.unsqueeze(-1))  #s.unsqueeze(-1).size() = (batch,1,dim,1)
        new_s = new_s.squeeze(-1) #now it is (batch,1,dim)

        d = dist(new_s,o)
        
        #new_loss = cross_entropy(-d,target)

        new_loss = th.nn.functional.cross_entropy(d,target, reduction='none')

        if_less = (min_loss > new_loss).long() #1 if new_loss is smaller, size: (batch)
                
        #prepare indexes for th.gather
        indexes = if_less.unsqueeze(-1)
        indexes = indexes.repeat(1,dim) #size: (batch,dim), value does not depend on dim
        indexes = indexes.unsqueeze(1) #size: (batch,1,dim)

        #concatenate two tensors, for each vertex in batch we need to choose one either from
        #the first of the second depending which loss was smaller
        centers = th.cat((min_centers.unsqueeze(1),new_s),1) #size: (batch, 2, dim)
        loss = th.cat((min_loss.unsqueeze(-1),new_loss.unsqueeze(-1)),1) #size: (batch, 2)

        min_centers = th.gather(centers, 1, indexes).squeeze() #selects the minimum, size: (batch,dim)
        min_loss = th.gather(loss,1,if_less.unsqueeze(-1)).squeeze() #size: (batch)

        t = LorentzManifold.ldot(min_centers[0],min_centers[0])
        if t.item() != -1: print(min_centers[0], LorentzManifold.ldot(min_centers[0],min_centers[0]))

    return min_centers.requires_grad_(), th.mean(min_loss)


def cross_entropy2(d):
    '''
    Computes cross entropy for each vertex from a batch separately

    Args:
        d: (torch.tensor[batch,1+negs]): #batch vertices, inputs[:,0] = dist to a nbr, inputs[:,1:] = dist to non-nbrs. 
        return: (torch.tensor[batch]): cross entropy loss for each vector from a batch 
    '''

    exp_in = th.exp(d)
    denom = th.sum(exp_in, 1)
    Q = exp_in[:,0]/denom
    
    return -th.log(Q)

#instead of using LorentzManifold().distance one can use this one
def distance(u,v):
    '''
    dim is the dimension of the hyperbolic space + 1
    Note: similar as group_rie.GroupRieDistance but simpler.
    
    Args:
        u: (torch.tensor[batch, 1, dim]): #batch vectors, every repeated 1+negs times.  
        v: (torch.tensor[batch, 1+negs, dim]): for every index in the first component we have 1+negs vectors, first is connected to 
                                               respective vector in u, next 50 are not connected. 
        dist: (torch.tensor[batch, 1+negs]): for every vertex in batch we have distances to 1+negs vertices. 
    '''

    #print(u.size(),v.size())

    Luv = LorentzManifold.ldot(u,v) #broadcasting in the second coordinate of u

    #print(Luv.size())
    d = acosh(-Luv)

    return d

def acosh(L):
    invL = th.ones_like(L)/L
    return  th.log(L) + th.log(1 + th.sqrt(th.clamp(1-invL*invL,0)))