#!/usr/bin/env python3
from numpy.core.numeric import correlate
import gpt as g
import numpy as np
import sys

rng = g.random("test")
U = g.qcd.gauge.random(g.grid([16, 16, 16, 16], g.double), rng)
#U = g.load("/home/scior/converter/l6464f21b7130m00119m0322a.1056.nersc")

U_hyp = g.qcd.gauge.smear.hyp(U, alpha = np.array([0.75, 0.6, 0.3]))

plaq_hyp = g.qcd.gauge.plaquette(U_hyp)

g.message("###########MEMORY REPORT AFTER HYP")
g.mem_report()

g.message(f"HYP smeared Plaquette = {plaq_hyp}")

grid = U_hyp[0].grid

p = {
    "kappa" : 0.12623,
    "csw_r": 1.0372,
    "csw_t": 1.0372,
    "xi_0": 1,
    "nu": 1,
    "isAnisotropic": False,
    "boundary_phases": [1, 1, 1, -1],
}

g.message("###########MEMORY REPORT BEFORE DIRAC OPERATOR")
g.mem_report()
w = g.qcd.fermion.wilson_clover(U_hyp, p)
g.message("###########MEMORY REPORT AFTER DIRAC OPERATOR")
g.mem_report()
# create point source
src = g.mspincolor(grid)
g.create.point(src, [0, 0, 0, 0])

# even-odd preconditioned matrix
# build solver using g5m and cg
inv = g.algorithms.inverter
pc = g.qcd.fermion.preconditioner
cg = inv.cg({"eps": 1e-6, "maxiter": 1000})

slv = w.propagator(inv.preconditioned(pc.g5m_ne(), cg))

# propagator
dst = g.mspincolor(grid)
dst @= slv * src

#get # of iterations
# dc_iter = len(cg.history) 

# pion
corr_pion = g.slice(g.trace(g.adj(dst) * dst), 3)
print("Pion two point:")
print(corr_pion)

corr_a4pi = g.slice(g.trace(g.adj(dst) * g.gamma["T"] * dst),3)

print("A4-pi two point:")
print(corr_a4pi)


#Make Pion DA with offset z in z-direction
z = 4
#and boost
k = 2.0

#Wilson line from 0 -> z
W = U_hyp[0][0,0,0,0]

U_prime, trafo = g.gauge_fix(U_hyp)

for dz in range(z):
    W =  U_hyp[2][0,0,dz,0] * W


# create smeared source, with some kind of smearing e.g.
#sm_src = g.create.smear.boosted_smearing(trafo, src, w=1.0, boost=[0.0,0.0,k])
#smear = g.create.smear.gauss(U_hyp, sigma=0.5, steps=3, dimensions=[0, 1, 2])
#sm_src = g(smear*src)
sm_src = src

src2 = g.mspincolor(grid)
g.create.point(src, [0, 0, z, 0])
#also here one could do some form of smearing

dst @= slv * sm_src

dst2 = g.mspincolor(grid)
dst2 @= slv * src2

#apply boosted smearing to the propagator
sm_dst = g.create.smear.boosted_smearing(trafo, dst, w=1.0, boost=[0.0,0,0,k])

sm_dst2 = g.create.smear.boosted_smearing(trafo, dst2, w=1.0, boost=[0.0,0,0,k])

# momentum
p = 2.0 * np.pi * np.array([0, 0, int(k), 0]) / L
P = g.exp_ixp(p)

correlator = g.slice(g.trace(g.adj(w) *  g.gamma[5] * g.adj(sm_dst2) * sm_dst * P * g.gamma["Z"] * g.gamma[5]) ,3)


# for t_op in range(T):
 
#     src_seq[:] = 0
#     src_seq[:, :, :, t_op] = sm_dst[:, :, :, t_op]
#     src_seq @= G_op * src_seq
#     sm_src_seq = g.create.smear.boosted_smearing(trafo, dst, w=1.0, boost=[0.0,0,0,k])
#     dst_seq @= slv * sm_src_seq
#     #F = g(smear* dst_seq) #using gaussian smearing on point src
#     F = dst_seq #using no smearin on point src

#     correlator.append(g.trace( g.adj(W) * g.gamma["Z"] * g.gamma[5] * F[0,0,z,0]))