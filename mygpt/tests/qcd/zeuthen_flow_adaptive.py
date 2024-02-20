import gpt as g


grid = g.grid([8,8,8,8], g.double)

rng = g.random('test')

U = g.qcd.gauge.random(grid,rng)
plaq = g.qcd.gauge.stencil.plaquette(U)

g.message("Initial plaquette: ", plaq)
epsilon = 0.001
maxtau = 2
g.message("Starting zeuthen flow with adaptive stepsize initial eps = ", epsilon)

U_flowed = g.qcd.gauge.smear.zeuthen_flow_gauge.zeuthen_flow_gauge_adaptive(U,epsilon,maxTau=maxtau, meas_interval=0)

plaq = g.qcd.gauge.stencil.plaquette(U_flowed)

g.message(f"Plaquette after zeuthen flow: {plaq}")

assert(abs(plaq-1.0) < 1e-5)
g.message("Test passed!")