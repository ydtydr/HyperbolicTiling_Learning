import torch as th
import numpy as np
from hype.lorentz import LorentzManifold as L

def vinberg17():
	'''Vinberg non-compact polyhedron in 17-dim Lobachevsky (Lorentz) space.  
	22 vectors r (roots of a polyhedron),
	R[i] is a Lobachevsky reflection through a wall perpendicular to r[i] multiplied by the norm of r[i]
	Thus R[i]/r[i] is the reflection 
	Only R[18]/r[18] is not integral, but r[18] = 10

	Returns R[i] and lists of norms of r
	'''
	
	#versors in R^18
	ver = th.zeros(18,18)
	for i in range(18): ver[i,i] = 1

	#vectors ortogonal to faces
	r = th.zeros(22,18)

	#defining 22 vectors 
	#Table 4 on page 31 of Vinberg's ON GROUPS OF UNIT ELEMENTS OF CERTAIN QUADRATIC FORMS
	#there is a mistake in the paper, the n+2 vector should be 3v_0 + v_1 + ... + v_11
	for i in range(16): r[i] = -ver[i+1]+ver[i+2]

	r[16] = -ver[17]

	r[17] = ver.narrow(0,0,4).sum(0)

	r[18] = ver.narrow(0,0,12).sum(0) + 2*ver[0]

	r[19] = ver.narrow(0,0,16).sum(0)+3*ver[0]+ver[1]

	r[20] = ver.narrow(0,0,18).sum(0)+3*ver[0]

	r[21] = 6*ver[0]+2*ver.narrow(0,1,7).sum(0)+ver.narrow(0,8,10).sum(0)

	#R[i] = norm of r[i] squared * transpose of a reflection matrix through a hyperplane ortogonal to r[i]
	R = th.zeros(22,18,18)
	RT = th.zeros(22,18,18)
	r_norm = th.zeros(22)

	for i in range(22): 
	    r_norm[i] = L.ldot(r[i],r[i])
	    ldot = L.ldot(ver,r[i]).unsqueeze(-1)
	    R[i] = r_norm[i]*ver - 2*ldot*r[i]
	    RT[i] = R[i].t()

	return RT, r_norm, r

def bugaenko6():

	a_cons = th.zeros(2)
	a_cons[0] = 1
	a_cons[1] = 1

	f = open("hype/bugaenko6.txt", "r")

	x = []
	
	for line in f:
		l = line[:-1].split(',')
		for j in range(len(l)):
			if l[j].isdigit(): l[j] = int(l[j])
			else: l[j] = -1
		x.append(l)   #list of 34 vectors, each 14 digits  	

	#this is a list of 34 vectors, each containing 7 coordinates 
	r = th.zeros(2,34,7)

	for i in range(34):
		for j in range(7):
			r[0,i,j]= x[i][2*j] 
			r[1,i,j]= x[i][2*j+1]

	R = th.zeros(2,34,7,7)
	RT = th.zeros(2,34,7,7)
	r_norm = th.zeros(2,34)

	#versors in R^7
	ver = th.zeros(2,7,7)
	for j in range(7): ver[0,j,j] = 1

	for i in range(34): 
		ldot = th.zeros(2,7)
		r_norm[0][i], r_norm[1][i] = L.const_ldot(r[:,i,:] , r[:,i,:] , a_cons)

		for j in range(7): 
			ldot[0,j], ldot[1,j] = L.const_ldot(ver[:,j,:] , r[:,i,:] , a_cons)
		
		ldot.unsqueeze_(-1)

		R[0,i] = r_norm[0,i]*ver[0] - 2*(ldot[0]*r[0,i] + 2*ldot[1]*r[1,i])
		R[1,i] = r_norm[1,i]*ver[0] - 2*(ldot[1]*r[0,i] + ldot[0]*r[1,i])

		RT[0,i] = R[0,i].t()
		RT[1,i] = R[1,i].t()

	return RT, r_norm, r