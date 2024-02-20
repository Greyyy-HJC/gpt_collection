import datetime
import h5py
import argparse
import re
import os
import subprocess
import gpt as g
import numpy as np
import sys
import copy
from operator import itemgetter
import sys, cgpt
from collections import defaultdict

parameters = {
    "placeholder" : [0]
}


def main():
    if g.rank() == 0:
        g.message(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
    PathConf = g.default.get("--PathConf", "PathConf")
    PathTwoPtFolder = g.default.get("--PathTwoPtFolder", "PathTwoPtFolder")
    PathTmunuOutFolder = g.default.get("--PathTmunuOutFolder", "PathTmunuOutFolder")
    confnum = g.default.get("--confnum", "confnum")
    trans_pmax = g.default.get_ivec("--trans_pmax", None, 4)
    U = g.convert(g.load(PathConf), g.double)
    g.message("finished loading gauge config")
    grid = U[0].grid
    L = U[0].grid.gdimensions
    Ns = U[0].grid.gdimensions[0]
    coor=g.coordinates(U[0])
    objects = g.qcd.gauge.get_gluonic_objects(parameters)

    if trans_pmax[3] != 0:
        g.message("momentum in t direction must be 0. exit.")
        sys.exit(1)

    t2E = []
    TmunuSave = []
    n_flow = 100 #t/a^2 = 1/8, 4/8, 9/8, 16/8 for smearing radius = a, 2a, 3a, 4a
    epsilon=0.05

    ft = 0
    for i_ft in range(n_flow):
        print("i_ft: ",i_ft)
        E = objects.get_gluon_anomaly(U)
        t2E.append([ft, np.sum(g.slice(E, 3)).real/float(L[0]*L[1]*L[2]*L[3])])
        Umunu = objects.get_Umunu(U)
        for px in range(-trans_pmax[0], trans_pmax[0]+1):
            for py in range(-trans_pmax[1], trans_pmax[1]+1):
                for pz in range(-trans_pmax[2], trans_pmax[2]+1):
                    temp=[]
                    insert_phase_factor = g.exp_ixp(2.0 * np.pi * np.array([px,py,pz,0]) / L)
                    E_with_phase = insert_phase_factor*E
                    temp.append(g.slice(E_with_phase, 3))
                    for ii in range(len(Umunu)):
                        Umunu_with_phase = insert_phase_factor*Umunu[ii]
                        temp.append(g.slice(Umunu_with_phase, 3))
                    TmunuSave.append(np.array(temp))
        if i_ft == 0:
            U = g.zeuthen_flow_gauge_fixedstepsize(U, epsilon=1e-4, Nstep=10, meas_interval=10)
            ft += 1e-4*10
            U = g.zeuthen_flow_gauge_fixedstepsize(U, epsilon=1e-3, Nstep=9, meas_interval=9)
            ft += 1e-3*9
            U = g.zeuthen_flow_gauge_fixedstepsize(U, epsilon=1e-2, Nstep=9, meas_interval=9)
            ft += 1e-2*9
        else:
            U = g.zeuthen_flow_gauge_fixedstepsize(U, epsilon=epsilon, Nstep=1, meas_interval=1)
            ft += epsilon
    TmunuSave=np.array(TmunuSave)
    TmunuSave=TmunuSave.reshape(n_flow, int((trans_pmax[0]*2+1)*(trans_pmax[1]*2+1)*(trans_pmax[2]*2+1)), 11, L[3])

    write = h5py.File(PathTmunuOutFolder+'/Tmunu_conf'+str(confnum)+'.hdf5',"w")
    for i_ft in range(n_flow):
        index = 0
        for px in range(-trans_pmax[0], trans_pmax[0]+1):
            for py in range(-trans_pmax[1], trans_pmax[1]+1):
                for pz in range(-trans_pmax[2], trans_pmax[2]+1):
                    write['ift_'+str(i_ft)+'_transpx'+str(px)+'_transpy'+str(py)+'_transpz'+str(pz)] = np.array(TmunuSave[i_ft,index,:,:])
                    index += 1

    t2E = np.array(t2E)
    np.savetxt(PathTmunuOutFolder+"/Tmunu_conf"+str(confnum)+"_t2E.dat", np.column_stack((t2E[:,0], t2E[:,0]**2*t2E[:,1])), header="Nft x (t, t2E)", fmt='%16.15e')

if __name__ == '__main__':
    main()
