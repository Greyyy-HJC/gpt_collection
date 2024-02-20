# %%
import os

os.sched_setaffinity(0, {0})  # just use one cpu

import gpt as g

rng = g.random("T")

def coulomb_gauge_fixing(U_read, eps, step, miter):
    # split in time
    Nt = U_read[0].grid.gdimensions[3]
    g.message(f"Separate {Nt} time slices")
    Usep = [g.separate(u, 3) for u in U_read[0:3]]
    Vt = [g.mcolor(Usep[0][0].grid) for t in range(Nt)]

    # optimizer
    opt = g.algorithms.optimize
    cg = opt.non_linear_cg(
                maxiter=miter, eps=eps, step=step
            )

    gd = opt.gradient_descent(maxiter=miter, eps=eps, step=step)  

    # Coulomb functional on each time-slice
    for t in range(Nt):
        f = g.qcd.gauge.fix.landau([Usep[mu][t] for mu in range(3)])
        fa = opt.fourier_accelerate.inverse_phat_square(Vt[t].grid, f)

        g.message(f"Run local time slice {t} / {Nt}")

        rng.element(Vt[t])

        # if not cg(fa)(Vt[t], Vt[t]):
        gd(fa)(Vt[t], Vt[t])

    Vt = [g.project(vt, "defect") for vt in Vt]

    # merge time slices
    V = g.merge(Vt, 3)
    U_fixed = g.qcd.gauge.transformed(U_read, V)

    return U_fixed



def fix_gauge(U, eps, step, gauge):
    """
    Input a gauge field U, output the gauge field after gauge fixing
    """
    V0 = g.identity(U[1])
    rng.element(V0)

    if gauge == "coulomb":
        # c = g.qcd.gauge.fix.coulomb(U)
        c = g.gauge_fix(U, maxiter=100, prec=1e-6)

    elif gauge == "landau":
        c = g.qcd.gauge.fix.landau(U)

    fac = g.algorithms.optimize.fourier_accelerate.inverse_phat_square(V0.grid, c)

    V1 = g.copy(V0)

    opt = g.algorithms.optimize
    cg = opt.non_linear_cg(maxiter=100, eps=eps, step=step)  # todo
    gd = opt.gradient_descent(maxiter=100, eps=eps, step=0.01)
    # if not cg(fac)([V1], [V1]):
    #     gd(fac)([V1], [V1])

    gd(fac)([V1], [V1])

    U_fixed = g.qcd.gauge.transformed(U, V1)

    return U_fixed


# %%
import gpt as g
rng = g.random("T")

U_ran = g.qcd.gauge.random(g.grid([8, 8, 8, 32], g.double), rng)

g.message(U_ran[0])

# U_fixed = fix_gauge(U_ran, 1e-8, 1e-4, "coulomb")


# %%
U_fixed_ls = []
for n in [4314, 4320, 4326, 4332, 4338]:
    filename = f"l4864f21b7373m00125m0250a.{n}.nersc"
    U_read = g.load(f"configs/NERSC/{filename}")
    eps = 1e-8
    step = 1e-4
    U_fixed = fix_gauge(U_read, eps, step, "landau")
    U_fixed_ls.append(U_fixed)



# # %%
# import os

# os.sched_setaffinity(0, {0})  # just use one cpu

# import gpt as g

# U_read = g.load("configs/l4864f21b7373m00125m0250a.4314.nersc")
# g.message(U_read[0])

# %%
import numpy as np
import gpt as g

rng = g.random("T")


for n in [4314, 4320, 4326, 4332, 4338]:
    filename = f"l4864f21b7373m00125m0250a.{n}.nersc"
    U_read = g.load(f"configs/NERSC/{filename}")

    opt = g.algorithms.optimize.non_linear_cg(maxiter=100, eps=1e-9, step=0.1)
    V0 = g.identity(U_read[0])
    rng.element(V0) #* random initial guess of the gauge transformation

    # Quark and solver setup (same for all source positions)
    grid = U_read[0].grid
    L = np.array(grid.fdimensions)

    w = g.qcd.fermion.wilson_clover(
    U_read,
    {
        "kappa": 0.137,
        "csw_r": 0,
        "csw_t": 0,
        "xi_0": 1,
        "nu": 1,
        "isAnisotropic": False,
        "boundary_phases": [1.0, 1.0, 1.0, 1.0],
    },
    )
    inv = g.algorithms.inverter
    pc = g.qcd.fermion.preconditioner
    cg = inv.cg({"eps": 1e-6, "maxiter": 1000})
    propagator = w.propagator(inv.preconditioned(pc.eo1_ne(), cg))

    # momentum
    p = 2.0 * np.pi * np.array([1, 0, 0, 0]) / L
    P = g.exp_ixp(p)

# Source positions
    src = g.mspincolor(grid)
    g.create.point(src, [0, 0, 0, 0])
    dst = g.mspincolor(grid)
    dst @= propagator * src

    # operators
    G_src = g.gamma[5] * P
    G_snk = g.gamma[5] * g.adj(P)

    # 2pt
    correlator_2pt = g.slice(g.trace(G_src * g.gamma[5] * g.adj(dst) * g.gamma[5] * G_snk * dst), 3)

    print(correlator_2pt)
# %%
