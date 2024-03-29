#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Production code to generate coarse-grid eigenvectors using existing
# fine-grid basis vectors
#
import gpt as g

# show available memory
g.mem_report()

# parameters
fn = g.default.get("--params", "params.txt")
params = g.params(fn, verbose=True)

# load configuration
U = params["config"]

# show available memory
g.mem_report()

# fermion
q = params["fmatrix"](U)

# load basis vectors
basis, feval = g.load(params["basis"])
nbasis = len(basis)

# memory info
g.mem_report()

# norms
for i in range(nbasis):
    g.message("Norm2 of basis[%d] = %g" % (i, g.norm2(basis[i])))

g.mem_report()

# prepare and test basis
for i in range(nbasis):
    g.message(i)
    _, eps2 = g.algorithms.eigen.evals(q.Mpc, [basis[i]], real=True)
    assert all([e2 < 1e-4 for e2 in eps2])
    g.mem_report(details=False)

# coarse grid
cgrid = params["cgrid"](q.Mpc.grid[0])
b = g.block.map(cgrid, basis)

# cheby on coarse grid
cop = params["cmatrix"](q.Mpc, b)

# implicitly restarted lanczos on coarse grid
irl = params["method_evec"]

# start vector
cstart = g.vcomplex(cgrid, nbasis)
cstart[:] = g.vcomplex([1] * nbasis, nbasis)

g.mem_report()

# basis
northo = params["northo"]
for i in range(northo):
    g.message("Orthonormalization round %d" % i)
    b.orthonormalize()

g.mem_report()

# now define coarse-grid operator
g.message(
    "Test precision of promote-project chain: %g"
    % (g.norm2(cstart - b.project * b.promote * cstart) / g.norm2(cstart))
)

g.mem_report()

try:
    cevec, cev = g.load("cevec")
except g.LoadError:
    cevec, cev = irl(cop, cstart, params["checkpointer"])
    g.save("cevec", (cevec, cev))

# smoother
smoother = params["smoother"](q.Mpc)
nsmoother = params["nsmoother"]
v_fine = g.lattice(basis[0])
v_fine_smooth = g.lattice(basis[0])
try:
    ev3 = g.load("ev3")
except g.LoadError:
    ev3 = [0.0] * len(cevec)
    for i, v in enumerate(cevec):
        v_fine @= b.promote * v
        for j in range(nsmoother):
            v_fine_smooth @= smoother * v_fine
            v_fine @= v_fine_smooth / g.norm2(v_fine_smooth) ** 0.5
        ev_smooth, ev_eps2 = g.algorithms.eigen.evals(q.Mpc, [v_fine], real=True)
        assert ev_eps2[0] < 1e-2
        ev3[i] = ev_smooth[0]
        g.message("Eigenvalue %d = %.15g" % (i, ev3[i]))
    g.save("ev3", ev3)

# tests
start = g.lattice(basis[0])
start[:] = g.vspincolor([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]])
start *= 1.0 / g.norm2(start) ** 0.5


def save_history(fn, history):
    f = open(fn, "wt")
    for i, v in enumerate(history):
        f.write("%d %.15E\n" % (i, v))
    f.close()


test_solver = params["test_solver"]
solver = g.algorithms.inverter.sequence(
    g.algorithms.inverter.coarse_deflate(cevec, basis, ev3), test_solver
)(q.Mpc)
v_fine[:] = 0
solver(v_fine, start)
save_history("cg_test.defl_all_ev3", test_solver.history)

solver = g.algorithms.inverter.sequence(
    g.algorithms.inverter.coarse_deflate(cevec[0 : len(basis)], basis, ev3[0 : len(basis)]),
    params["test_solver"],
)(q.Mpc)
v_fine[:] = 0
solver(v_fine, start)
save_history("cg_test.defl_full", test_solver.history)

v_fine[:] = 0
test_solver(q.Mpc)(v_fine, start)
save_history("cg_test.undefl", test_solver.history)

# save in rbc format
g.save("lanczos.output", [basis, cevec, ev3], params["format"])
