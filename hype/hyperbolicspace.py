#!/usr/bin/env python3
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch as th
from torch.autograd import Function
from .common import acosh
from .manifold import Manifold
import numpy as np
import hype.reflection_sets as reflection_sets

def move_to_hyperboloid(g,a=1):
    g[0] = th.sqrt((1/a)*(1+(g[1:]*g[1:]).sum()))
    return g

def normalize_hyperbolicspace(g,g_matrix, polytope, dim, debug=0):
    
    #dim is the dimension of the polytope+1

    R, norms, r, x0 = getattr(reflection_sets,polytope)(dim-1)

    #if opt.manifold == 'hyperbolic_cube': R, norms, r, x0 = reflection_sets.hyperbolic_cube(opt.dim-1)
    #if opt.manifold == 'vinberg17': R, norms, r, x0 = reflection_sets.vinberg17()
    #if opt.manifold == 'vinberg3': R, norms, r, x0 = reflection_sets.vinberg3()

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
        #at least arccosh(2)/100 (for stability) and g is on the other side of the hyperplane than the polytope. 

        max_index = -1
        max_lprod = -1

        for i in range(number_of_roots):
            lprod = HyperbolicSpace.ldot(RVIg,r[i])
            if lprod >= norms[i]/100 and lprod > max_lprod: 
                max_lprod = lprod
                max_index = i
        
        if max_index == -1: break
        else:
            RV = th.matmul(RV,R[max_index])
            RVI = th.matmul(R[max_index],RVI)
            RVIg = th.matmul(RVI,g)
            RVIg = move_to_hyperboloid(RVIg)

    if debug==1:    
        print('RV', RV)

        for i in range(number_of_roots):
            print('<r,g>', HyperbolicSpace.ldot(r[i],RVIg).item() < norms[i])
    
    #another optimisation technique: just take the first reflection that makes g closer to the polytope. 
    '''    
    shorter = True
    #j=0

    while shorter:
        shorter = False
        i = 0
        #j = j+1

        lprod = LorentzManifold.ldot(RVIg,r[i])

        while 0 >= lprod: #put lprod <= norms[i] if there is a stability problem or any other epsilon in place of norms
            i = i+1
            if i == number_of_roots: break
            lprod = LorentzManifold.ldot(RVIg,r[i])
        else:
            RV = th.matmul(RV,R[i])
            RVI = th.matmul(R[i],RVI)
            RVIg = th.matmul(RVI,g)
            RVIg = move_to_hyperboloid(RVIg)
            shorter = True
        
    #print('RV', RV)
    #print('NSorm', Norm)
    #if i<34: print(R[:,i], j)
    #check distance of RVIg to x_0
    if debug==1: print(np.arccosh(-LorentzManifold.ldot(x0,RVIg)).item(), RVIg.max().item(), max_index)
    #, RVI[0].max().item(), RVI[1].max().item(), i, j)
    #print(RVIg, j)
    '''

    return RVIg, th.matmul(g_matrix, RV)

class HyperbolicSpace(Manifold):
    __slots__ = ["eps", "_eps", "norm_clip", "max_norm", "debug"]

    @staticmethod
    def dim(dim):
        #dim = the dimension of the polytope
        return dim+1 

    def __init__(self, eps=1e-12, _eps=1e-5, norm_clip=1, max_norm=1e6,
            debug=False, **kwargs):
        self.eps = eps
        self._eps = _eps
        self.norm_clip = norm_clip
        self.max_norm = max_norm
        self.debug = debug

    @staticmethod
    def ldot(u, v, keepdim=False):
        """Lorentzian Scalar Product"""
        uv = u * v
        uv.narrow(-1, 0, 1).mul_(-1)
        return th.sum(uv, dim=-1, keepdim=keepdim)    

    def to_poincare_ball(self, u, u_int_matrix):
        u = th.matmul(u_int_matrix, u.unsqueeze(-1)).squeeze(-1)
        d = u.size(-1) - 1
        return u.narrow(-1, 1, d) / (u.narrow(-1, 0, 1) + 1)

    def distance(self, uu, uu_int_matrix, vv, vv_int_matrix, g):
        dis = GroupRieDistance.apply(uu, uu_int_matrix, vv, vv_int_matrix, g)
        return dis

    def pnorm(self, u, u_int_matrix):
        return th.sqrt(th.sum(th.pow(self.to_poincare_ball(u, u_int_matrix), 2), dim=-1))

    def normalize(self, w):
        """Normalize vector such that it is located on the hyperboloid (same as move_to_hyperboloid)"""
        d = w.size(-1) - 1
        narrowed = w.narrow(-1, 1, d)
        if self.max_norm:
            narrowed.view(-1, d).renorm_(p=2, dim=0, maxnorm=self.max_norm)
        tmp = 1 + th.sum(th.pow(narrowed, 2), dim=-1, keepdim=True)
        tmp.sqrt_()
        w.narrow(-1, 0, 1).copy_(tmp)
        return w

    def normalize_tan(self, x_all, v_all): #to do, it is used in log?
        d = v_all.size(1) - 1
        x = x_all.narrow(1, 1, d)
        xv = th.sum(x * v_all.narrow(1, 1, d), dim=1, keepdim=True)
        tmp = 1 + th.sum(th.pow(x_all.narrow(1, 1, d), 2), dim=1, keepdim=True)
        tmp.sqrt_().clamp_(min=self._eps)
        v_all.narrow(1, 0, 1).copy_(xv / tmp)
        return v_all

    def init_weights(self, w, irange=1e-5): 
        w.data.uniform_(-irange, irange)
        w.data[...,0] = th.sqrt(th.clamp(th.sum(w[...,1:] * w[...,1:], dim=-1),min=0) + 1)

    def init_weights_int_matrix(self, w, faraway, dim, polytope):
        if not faraway: 
            ID = th.zeros_like(w[0])
            for i in range(w.size(-1)): ID[i,i] = 1
            w.data.zero_()
            w.data.add_(ID)
        else: 
            #creating a faraway point
            g = faraway*th.randn(dim) 
            g = move_to_hyperboloid(g) 

            G = th.eye(dim,dim) #G: matrix of the lorentz product
            G[0,0] = -1

            g0, RV = normalize_hyperbolicspace(g,G,polytope,dim)
            w.data.zero_()
            w.data.add_(RV)

        #vinberg3:
        #ID =  th.tensor([[ 521706., -363309., -303223., -219635.], 
        #              [ -11839.,    8244.,    6881.,    4985.],
        #              [ 339483., -236411., -197313., -142920.],
        #              [    -395965.,  275745.,  230140.,  166699.]])

        #vinberg17:
        #v17_faraway_tile_pd = pd.read_csv('../ipynb/vinberg17_fawaway_tile.csv')
        #read_tensor = th.tensor(read.to_numpy())


    def rgrad(self, p, d_p):
        """Riemannian gradient for hyperboloid"""
        if d_p.is_sparse:
            u = d_p._values()
            x = p.index_select(0, d_p._indices().squeeze())
        else:
            u = d_p
            x = p
        u.narrow(-1, 0, 1).mul_(-1)
        u.addcmul_(self.ldot(x, u, keepdim=True).expand_as(x), x)
        return d_p

    def expm(self, p, d_p, lr=None, out=None, normalize=False):
        """Exponential map for hyperboloid"""
        if out is None:
            out = p
        if d_p.is_sparse:
            ix, d_val = d_p._indices().squeeze(), d_p._values()
            p_val = self.normalize(p.index_select(0, ix))
            ldv = self.ldot(d_val, d_val, keepdim=True)
            if self.debug:
                assert all(ldv > 0), "Tangent norm must be greater 0"
                assert all(ldv == ldv), "Tangent norm includes NaNs"
            nd_p = ldv.clamp_(min=0).sqrt_()
            t = th.clamp(nd_p, max=self.norm_clip)
            nd_p.clamp_(min=self.eps)
            newp = (th.cosh(t) * p_val).addcdiv_(th.sinh(t) * d_val, nd_p)
            if normalize:
                newp = self.normalize(newp)
            p.index_copy_(0, ix, newp)
        else:
            if lr is not None:
                d_p.narrow(-1, 0, 1).mul_(-1)
                d_p.addcmul_((self.ldot(p, d_p, keepdim=True)).expand_as(p), p)
                d_p.mul_(-lr)
            ldv = self.ldot(d_p, d_p, keepdim=True)
            if self.debug:
                assert all(ldv > 0), "Tangent norm must be greater 0"
                assert all(ldv == ldv), "Tangent norm includes NaNs"
            nd_p = ldv.clamp_(min=0).sqrt_()
            t = th.clamp(nd_p, max=self.norm_clip)
            nd_p.clamp_(min=self.eps)
            newp = (th.cosh(t) * p).addcdiv_(th.sinh(t) * d_p, nd_p)
            if normalize:
                newp = self.normalize(newp)
            p.copy_(newp)

    def logm(self, x, y):
        """Logarithmic map on the Lorenz Manifold"""
        xy = th.clamp(self.ldot(x, y).unsqueeze(-1), max=-1)
        v = acosh(-xy, self.eps).div_(
            th.clamp(th.sqrt(xy * xy - 1), min=self._eps)
        ) * th.addcmul(y, xy, x)
        return self.normalize_tan(x, v)

    def ptransp(self, x, y, v, ix=None, out=None):
        """Parallel transport for hyperboloid"""
        if ix is not None:
            v_ = v
            x_ = x.index_select(0, ix)
            y_ = y.index_select(0, ix)
        elif v.is_sparse:
            ix, v_ = v._indices().squeeze(), v._values()
            x_ = x.index_select(0, ix)
            y_ = y.index_select(0, ix)
        else:
            raise NotImplementedError
        xy = self.ldot(x_, y_, keepdim=True).expand_as(x_)
        vy = self.ldot(v_, y_, keepdim=True).expand_as(x_)
        vnew = v_ + vy / (1 - xy) * (x_ + y_)
        if out is None:
            return vnew
        else:
            out.index_copy_(0, ix, vnew)

class GroupRieDistance(Function):
    @staticmethod
    def forward(self, u, u_int_matrix, v, v_int_matrix, g, AvOverflow = False, myeps1 = 1e-8 ,myeps2 = 1e-16, decompose_factor = 25):
        # decompose_factor = 11 for float32; decompose_factor = 25 for float64.
        assert th.isnan(u_int_matrix).max()==0, "u includes NaNs"
        assert th.isnan(v_int_matrix).max()==0, "v includes NaNs"
        #assert th.isnan(u_int_norm).max()==0, "u includes NaNs"
        #assert th.isnan(v_int_norm).max()==0, "v includes NaNs"

        #u: (batch, 1+negs, dim) - this is only #batch vectors, every repeated 1+negs times.  
        #v: (batch, 1+negs, dim) - for every index in the first component we have 1+negs vectors, first is connected to 
        #                          respective vector in u, next 50 are not connected. 

        if len(u)<len(v):
            u = u.expand_as(v)
            u_int_matrix = u_int_matrix.expand_as(v_int_matrix)
            #u_int_norm = u_int_norm.expand_as(v_int_norm)
        elif len(u)>len(v):
            v = v.expand_as(u)
            v_int_matrix = v_int_matrix.expand_as(u_int_matrix)
            #v_int_norm = v_int_norm.expand_as(u_int_norm)

        self.save_for_backward(u, v)
        ############# use U = U1+U2 version, we separate U^TM3V into (U1+U2)^TM3(V1+V2)=U1^TM3V1+U1^TM3V2+U2^TM3V1+U2^TM3V2,
        ############# in order to avoid numerical inprecision of storing
        ############# integers in float, and multiply them to get the other intergers, which may be incorrect due to inprecision.

        gv_int_matrix = th.matmul(g,v_int_matrix)

        #print(gv_int_matrix)

        u_int_matrix_trans = u_int_matrix.transpose(-2,-1)
        
        '''
        u_int_matrix2 = th.fmod(u_int_matrix_trans, 2 ** decompose_factor)
        u_int_matrix1 = u_int_matrix_trans - u_int_matrix2
        gv_int_matrix2 = th.fmod(gv_int_matrix, 2 ** decompose_factor)
        gv_int_matrix1 = gv_int_matrix - gv_int_matrix2
        '''

        #Q = th.zeros_like(v_int_matrix) #Q = U^TgV
        #N = th.zeros_like(u_int_norms)

        #for i in range(v_int_matrix.size(0)): #batch size
        #    for j in range(v_int_matrix.size(1)): #nnegs + 2
        
        Q = th.matmul(u_int_matrix_trans, gv_int_matrix)

        '''
        Q = th.matmul(u_int_matrix1, gv_int_matrix1)\
            +th.matmul(u_int_matrix1, gv_int_matrix2)\
            +th.matmul(u_int_matrix2, gv_int_matrix1)\
            +th.matmul(u_int_matrix2, gv_int_matrix2)
        '''
        #print('Q', Q.size())

        absQ = Q.abs()

        max_coef = absQ.max(-1, keepdim=True)[0].max(-2,keepdim=True)[0]

        self.hatQ = -th.div(Q,max_coef)

        #print(self.hatQ)
        #print(u[0,0],v[0,0])

        d_c = th.matmul(u.unsqueeze(-1).transpose(-2,-1), th.matmul(self.hatQ, v.unsqueeze(-1))).squeeze(-1).squeeze(-1)#cpu float       
        #print(d_c[5][5].item())
        inv_max_coef = th.div(th.ones_like(max_coef.squeeze(-1).squeeze(-1)),max_coef.squeeze(-1).squeeze(-1))
        self.nomdis = th.sqrt(th.clamp(d_c**2-inv_max_coef**2,min=myeps2))#cpu float
        #outp = arccosh(d_c) = log(d_c + sqrt(d_c^2 - 1))
        #print('nomdis', self.nomdis)
        outp = th.log(max_coef.squeeze(-1).squeeze(-1)) + th.log(th.clamp(d_c + self.nomdis,min=myeps1))#cpu float

        #print(d_c + self.nomdis)
        return outp

    @staticmethod
    def backward(self, g):
        u, v = self.saved_tensors
        g = g.unsqueeze(-1).expand_as(u)

        uupfrac = th.matmul(self.hatQ, v.unsqueeze(-1)).squeeze(-1)
        vupfrac = th.matmul(self.hatQ.transpose(-2,-1), u.unsqueeze(-1)).squeeze(-1)
        
        gu = th.div(uupfrac, self.nomdis.unsqueeze(-1).expand_as(uupfrac))
        gv = th.div(vupfrac, self.nomdis.unsqueeze(-1).expand_as(vupfrac))

        assert th.isnan(gu).max() == 0, "gu includes NaNs"
        assert th.isnan(gv).max() == 0, "gv includes NaNs"

        #print(gu)
        return g * gu, None, g * gv, None, None

#not used, it is now in discrete
class GroupRieDistanceDiscrete(Function):
    @staticmethod
    def forward(self, u, u_int_matrix, v, v_int_matrix, g, AvOverflow = False, myeps1 = 1e-8 ,myeps2 = 1e-16):
        ''' decompose_factor = 11 for float32; decompose_factor = 25 for float64.
        u: (batch, 1+negs, dim) - this is #batch vectors, every repeated 1+negs times.  
        v: (batch, 1+negs, dim) - for every index in the first component we have 1+negs vectors, first is connected to 
                                  respective vector in u, next 50 are not connected. 
        outp: (batch, 1+negs) - for every vertex in batch we have distances to 1+negs vertices.''' 

        assert th.isnan(u_int_matrix).max()==0, "u includes NaNs"
        assert th.isnan(v_int_matrix).max()==0, "v includes NaNs"
        
        gv_int_matrix = th.matmul(g,v_int_matrix)

        u_int_matrix_trans = u_int_matrix.transpose(-2,-1)
        
        Q = th.matmul(u_int_matrix_trans.narrow(2,0,1), gv_int_matrix.narrow(3,0,1))
        
        absQ = Q.abs()

        max_coef = absQ.max(-1, keepdim=True)[0].max(-2,keepdim=True)[0]

        self.hatQ = -th.div(Q,max_coef)

        d_c = self.hatQ.squeeze(-1).squeeze(-1)#cpu float       
        
        inv_max_coef = th.div(th.ones_like(max_coef.squeeze(-1).squeeze(-1)),max_coef.squeeze(-1).squeeze(-1))
        
        self.nomdis = th.sqrt(th.clamp(d_c**2-inv_max_coef**2,min=myeps2))#cpu float

        outp = th.log(max_coef.squeeze(-1).squeeze(-1)) + th.log(th.clamp(d_c + self.nomdis,min=myeps1))#cpu float

        return outp


    @staticmethod
    def backward(self, g):
        u, v = self.saved_tensors
        g = g.unsqueeze(-1).expand_as(u)

        uupfrac = self.hatQ.squeeze(-1)
        vupfrac = self.hatQ.transpose(-2,-1).squeeze(-1)
        
        gu = th.div(uupfrac, self.nomdis.unsqueeze(-1).expand_as(uupfrac))
        gv = th.div(vupfrac, self.nomdis.unsqueeze(-1).expand_as(vupfrac))

        assert th.isnan(gu).max() == 0, "gu includes NaNs"
        assert th.isnan(gv).max() == 0, "gv includes NaNs"

        return g * gu, None, g * gv, None, None