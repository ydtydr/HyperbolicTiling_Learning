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

class GroupRiehighManifold(Manifold):
    __slots__ = ["eps", "_eps", "norm_clip", "max_norm", "debug"]

    @staticmethod
    def dim(dim):
        return dim*3

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
        L = th.sqrt(th.Tensor([[3, 0, 0], [0, 1, 0], [0, 0, 1]]))
        R = th.sqrt(th.Tensor([[1.0 / 3.0, 0, 0], [0, 1, 0], [0, 0, 1]]))
        u = th.matmul(L, th.matmul(u_int_matrix, th.matmul(R, u.unsqueeze(-1)))).squeeze(-1)
        d = u.size(-1) - 1
        return u.narrow(-1, 1, d) / (u.narrow(-1, 0, 1) + 1)

    def distance(self, uu, uu_int_matrix, vv, vv_int_matrix):
        dimension = uu.size(-1)//3
        d_all = 0
        for i in range(dimension):
            d_all += GroupRiehighDistance.apply(uu[...,3*i:3*(i+1)], uu_int_matrix[...,i,:,:], vv[...,3*i:3*(i+1)], vv_int_matrix[...,i,:,:])
        return d_all
    
    def pnorm(self, u, u_int_matrix):
        dimension = u.size(-1)//3
        all_norm = 0
        for i in range(dimension):
            all_norm += th.sqrt(th.sum(th.pow(self.to_poincare_ball(u[...,3*i:3*(i+1)],u_int_matrix[...,i,:,:]), 2), dim=-1))
        return all_norm/dimension
    
    def normalize(self, ww, gra=True):
        """Normalize vector such that it is located on the hyperboloid"""
        if gra:
            dimension = ww.size(-1)//3
            for i in range(dimension):
                w = ww[...,3*i:3*(i+1)]
                d = w.size(-1) - 1
                narrowed = w.narrow(-1, 1, d)
                if self.max_norm:
                    narrowed.view(-1, d).renorm_(p=2, dim=0, maxnorm=self.max_norm)
                tmp = 1 + th.sum(th.pow(narrowed, 2), dim=-1, keepdim=True)
                tmp.sqrt_()
                w.narrow(-1, 0, 1).copy_(tmp)
        else:
            w = ww
            d = w.size(-1) - 1
            narrowed = w.narrow(-1, 1, d)
            if self.max_norm:
                narrowed.view(-1, d).renorm_(p=2, dim=0, maxnorm=self.max_norm)
            tmp = 1 + th.sum(th.pow(narrowed, 2), dim=-1, keepdim=True)
            tmp.sqrt_()
            w.narrow(-1, 0, 1).copy_(tmp)
        return ww
    

    def normalize_tan(self, x_all, v_all):
        d = v_all.size(1) - 1
        x = x_all.narrow(1, 1, d)
        xv = th.sum(x * v_all.narrow(1, 1, d), dim=1, keepdim=True)
        tmp = 1 + th.sum(th.pow(x_all.narrow(1, 1, d), 2), dim=1, keepdim=True)
        tmp.sqrt_().clamp_(min=self._eps)
        v_all.narrow(1, 0, 1).copy_(xv / tmp)
        return v_all
    
    def init_weights(self, w, irange=1e-5):
        w.data.uniform_(-irange, irange)
        w.data.copy_(self.normalize(w.data))

    def init_weights_int_matrix(self, w):
        ID = th.eye(3,3)
        w.data.zero_()
        w.data.add_(ID)
    
    def rgrad(self, p, d_p):
        """Riemannian gradient for hyperboloid"""
        if d_p.is_sparse:
            uu = d_p._values()
            xx = p.index_select(0, d_p._indices().squeeze())
        else:
            uu = d_p
            xx = p
        dimension = p.size(-1)//3
        for i in range(dimension):
            u = uu[...,3*i:3*(i+1)]
            x = xx[...,3*i:3*(i+1)]
            u.narrow(-1, 0, 1).mul_(-1)
            u.addcmul_(self.ldot(x, u, keepdim=True).expand_as(x), x)
        return d_p

    def expm(self, pp, d_pp, lr=None, out=None, normalize=False):
        dimension = pp.size(-1)//3
        ix, d_val_p = d_pp._indices().squeeze(), d_pp._values()
        p_val_p = self.normalize(pp.index_select(0, ix))
        for i in range(dimension):
            p = pp[...,3*i:3*(i+1)]
            d_val = d_val_p[...,3*i:3*(i+1)]
            p_val = p_val_p[...,3*i:3*(i+1)]
            ldv = self.ldot(d_val, d_val, keepdim=True)
            if self.debug:
                assert all(ldv > 0), "Tangent norm must be greater 0"
                assert all(ldv == ldv), "Tangent norm includes NaNs"
            nd_p = ldv.clamp_(min=0).sqrt_()
            t = th.clamp(nd_p, max=self.norm_clip)
            nd_p.clamp_(min=self.eps)
            newp = (th.cosh(t) * p_val).addcdiv_(th.sinh(t) * d_val, nd_p)
            if normalize:
                newp = self.normalize(newp,gra=False)
            p.index_copy_(0, ix, newp)

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

class GroupRiehighDistance(Function):
    @staticmethod
    def forward(self, u, u_int_matrix, v, v_int_matrix, AvOverflow=False, myeps1=1e-8, myeps2=1e-16,
                decompose_factor=25):
        # decompose_factor = 11 for float32; decompose_factor = 25 for float64.
        assert th.isnan(u_int_matrix).max() == 0, "u includes NaNs"
        assert th.isnan(v_int_matrix).max() == 0, "v includes NaNs"
        if len(u) < len(v):
            u = u.expand_as(v)
            u_int_matrix = u_int_matrix.expand_as(v_int_matrix)
        elif len(u) > len(v):
            v = v.expand_as(u)
            v_int_matrix = v_int_matrix.expand_as(u_int_matrix)
        self.save_for_backward(u, v)
        M3 = th.Tensor([[3, 0, 0], [0, -1, 0], [0, 0, -1]])
        R = th.sqrt(th.Tensor([[1.0 / 3.0, 0, 0], [0, 1, 0], [0, 0, 1]]))
        ############# use U = U1+U2 version, we separate U^TM3V into (U1+U2)^TM3(V1+V2)=U1^TM3V1+U1^TM3V2+U2^TM3V1+U2^TM3V2,
        ############# in order to avoid numerical inprecision of storing
        ############# integers in float, and multiply them to get the other intergers, which may be incorrect due to inprecision.
        u_int_matrix2 = th.fmod(u_int_matrix, 2 ** decompose_factor)
        u_int_matrix1 = u_int_matrix - u_int_matrix2
        v_int_matrix2 = th.fmod(v_int_matrix, 2 ** decompose_factor)
        v_int_matrix1 = v_int_matrix - v_int_matrix2
        Q = th.matmul(u_int_matrix1.transpose(-2, -1), th.matmul(M3, v_int_matrix1)) \
            + (th.matmul(u_int_matrix1.transpose(-2, -1), th.matmul(M3, v_int_matrix2))
               + th.matmul(u_int_matrix2.transpose(-2, -1), th.matmul(M3, v_int_matrix1))) \
            + th.matmul(u_int_matrix2.transpose(-2, -1), th.matmul(M3, v_int_matrix2))
        Q11 = th.clamp(Q.narrow(-2, 0, 1).narrow(-1, 0, 1), min=myeps1)  # divide Q by Q11 to avoid overflow
        if not AvOverflow:  #### if the dataset is not complex, and there is overflow concern, we set Q11=1, then Q=hatQ, if AvOverflow is false
            Q11 = th.clamp(Q11, max=1)
        self.hatQ = th.div(Q, Q11.expand_as(Q))  # divided by Q11
        RThatQR = th.matmul(R, th.matmul(self.hatQ, R))  # cpu float
        d_c = th.matmul(u.unsqueeze(-1).transpose(-2, -1), th.matmul(RThatQR, v.unsqueeze(-1))).squeeze(-1).squeeze(
            -1)  # cpu float
        invQ11 = th.div(th.ones_like(Q11.squeeze(-1).squeeze(-1)), Q11.squeeze(-1).squeeze(-1))  # cpu float
        self.nomdis = th.sqrt(th.clamp(d_c * d_c - invQ11 * invQ11, min=myeps2))  # cpu float
        outp = th.log(Q11.squeeze(-1).squeeze(-1)) + th.log(th.clamp(d_c + self.nomdis, min=myeps1))  # cpu float
        return outp

    @staticmethod
    def backward(self, g):
        R = th.sqrt(th.Tensor([[1.0 / 3.0, 0, 0], [0, 1, 0], [0, 0, 1]]))
        u, v = self.saved_tensors
        g = g.unsqueeze(-1).expand_as(u)
        uupfrac = th.matmul(R, th.matmul(self.hatQ, th.matmul(R, v.unsqueeze(-1)))).squeeze(-1)
        vupfrac = th.matmul(R, th.matmul(self.hatQ.transpose(-2, -1), th.matmul(R, u.unsqueeze(-1)))).squeeze(-1)
        gu = th.div(uupfrac, self.nomdis.unsqueeze(-1).expand_as(uupfrac))
        gv = th.div(vupfrac, self.nomdis.unsqueeze(-1).expand_as(vupfrac))
        assert th.isnan(gu).max() == 0, "gu includes NaNs"
        assert th.isnan(gv).max() == 0, "gv includes NaNs"
        return g * gu, None, g * gv, None