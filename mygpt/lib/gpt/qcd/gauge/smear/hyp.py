#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
import numpy as np
import gpt as g
from gpt.params import params_convention


# HYP smearing as in QLUA

@params_convention(alpha=None)
def hyp(U, params):

    alpha = params["alpha"]
    assert alpha is not None
    assert alpha.shape == (3,)

    U_hyp = []

    #lvl 1 smearing

    ftmp = alpha[2]/2.0/(1.0-alpha[2])

    V1 = {}
    
    Y = g.lattice(U[0])
    staple = g.lattice(U[0])

    for mu in range(4):
        V1[mu] = {}
        for nu in range(4):
            if(nu==mu):
                continue
            V1[mu][nu] = {}

    for mu in range(4):
        g.copy(Y,U[mu])

        for nu in range(4):
            if(nu==mu):
                continue
            for rho in range(nu+1,4):
                if(rho==mu or rho ==nu):
                    continue
                staple[:] = 0
                for sigma in [x for x in range(4) if x not in [mu,nu,rho]]:
                    Us_f = g.cshift(U[sigma], mu, 1)

                    staple += U[sigma] * g.cshift(U[mu], sigma, 1) * g.adj(Us_f)
                    staple += g.cshift(g.adj(U[sigma]) * U[mu] * Us_f, sigma, -1)

                X = g.eval(g.adj(U[mu] + ftmp * staple))

                g.projectSU3(X, Y)

                V1[mu][nu][rho] = g.lattice(U[0]) 
                V1[mu][rho][nu] = g.lattice(U[0])

                g.copy(V1[mu][nu][rho], Y)
                g.copy(V1[mu][rho][nu], Y)

    # lvl 1 complete

    # lvl 2 smearing

    ftmp = alpha[1]/4.0/(1.0-alpha[1])

    V2 = {}

    for mu in range(4):
        V2[mu] = {}
        g.copy(Y, U[mu])

        for nu in range(4):
            if(nu==mu):
                continue

            staple[:] = 0
            for sigma in range(4):
                if(sigma==mu or sigma==nu):
                    continue

                Vsmn_f = g.cshift(V1[sigma][mu][nu], mu, 1)
                staple += V1[sigma][mu][nu] * g.cshift(V1[mu][sigma][nu], sigma, 1) * g.adj(Vsmn_f)
                staple += g.cshift(g.adj(V1[sigma][mu][nu]) * V1[mu][sigma][nu] * Vsmn_f, sigma, -1)

            X = g.eval(g.adj(U[mu] + ftmp * staple))
            g.projectSU3(X, Y)

            V2[mu][nu] = g.lattice(U[0])
            g.copy(V2[mu][nu], Y)
    
    # lvl 2 complete

    # lvl 3 smearing

    ftmp = alpha[0]/6.0/(1.0-alpha[0])

    U_hyp = []

    for mu in range(4):
        g.copy(Y, U[mu])
        staple[:] = 0

        for sigma in range(4):
            if(sigma==mu):
                continue

            Vsm_f = g.cshift(V2[sigma][mu], mu, 1)

            staple += V2[sigma][mu] * g.cshift(V2[mu][sigma], sigma, 1) * g.adj(Vsm_f)
            staple += g.cshift(g.adj(V2[sigma][mu]) * V2[mu][sigma] * Vsm_f, sigma, -1)

        X = g.eval(g.adj(U[mu] + ftmp * staple))
        g.projectSU3(X, Y)

        U_hyp.append(g.lattice(U[0]))
        g.copy(U_hyp[mu], Y)

    
    return U_hyp




 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
    # V1 = {}

    # for mu in range(4):
    #     V1[mu] = {}
    #     for nu in range(4):
    #         V1[mu][nu] = {}

    # staple = g.lattice(U[0])
    # Y = g.lattice(U[0])


    # for mu in range(4):

    #     g.copy(Y, U[mu])

    #     for nu in range(4):

    #         if(nu==mu):
    #             continue

    #         # for rho in range(4):
    #         for rho in range (nu+1,4):

    #             if(rho==nu or rho==mu):
    #                 continue
       
    #             staple[:] = 0
    #             #sigma = [x for x in range(4) if x not in [mu,nu,rho]][0]
    #             for sigma in [x for x in range(4) if x not in [mu,nu,rho]]:
    #                 U_tmp = g.cshift(U[sigma], mu, 1)
    #                 staple += U[sigma]*g.cshift(U[mu], sigma, 1)*g.adj(U_tmp)
    #                 staple += g.cshift(g.adj(U[sigma])*U[mu]*U_tmp, sigma, -1)

                
    #             X = g.eval(g.adj(U[mu] + ftmp2*staple))
    #             g.projectSU3(X, Y)

    #             V1[mu][nu][rho] = g.lattice(U[0])
    #             V1[mu][rho][nu] = g.lattice(U[0])

    #             g.copy(V1[mu][nu][rho], Y)
    #             g.copy(V1[mu][nu][rho], Y)

    # #lvl 1 complete

    # #lvl 2 smearing

    # ftmp2 = alpha[1]/4.0/(1.0-alpha[1])

    # V2 = {}
    # for mu in range(4):
    #     V2[mu] = {}
    #     g.copy(Y,U[mu])

    # for mu in range(4):
    #     for nu in range(4):
    #         if(nu==mu):
    #             continue
    #         staple[:] = 0

    #         for sigma in range(4):
    #             if(sigma==nu or sigma==mu):
    #                 continue

    #             V_tmp = g.cshift(V1[sigma][mu][nu] ,mu, 1)
    #             staple += V1[sigma][mu][nu]*g.cshift(V1[mu][sigma][nu],sigma,1)*g.adj(V_tmp)
    #             staple += g.cshift(g.adj(V1[sigma][mu][nu])*V1[mu][sigma][nu]*V_tmp , sigma, -1)

    #         X = g.eval(g.adj(U[mu] + ftmp2*staple))
    #         g.projectSU3(X, Y)

    #         V2[mu][nu] = g.lattice(U[0])
    #         g.copy(V2[mu][nu], Y)

    # #lvl 2 complete

    # #lvl 3 smearing

    # ftmp2 = alpha[0]/6.0/(1.0-alpha[0])

    # for mu in range(4):
    #     g.copy(Y, U[mu])
    #     staple[:] = 0
    #     for sigma in range(4):
    #         if(sigma==mu):
    #             continue

    #         V_tmp = g.cshift(V2[sigma][mu], mu, 1)
    #         staple += V2[sigma][mu]*g.cshift(V2[mu][sigma], sigma, 1)*g.adj(V_tmp)
    #         staple += g.cshift(g.adj(V2[sigma][mu])*V2[mu][sigma]*V_tmp , sigma, -1)

    #     X = g.eval(g.adj(U[mu] + ftmp2*staple))
    #     g.projectSU3(X, Y)
    #     U_hyp.append(Y)

    
    # return U_hyp

    