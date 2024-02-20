import gpt as g


grid = g.grid([8,8,8,8], g.double)

rng = g.random('test')

U = g.qcd.gauge.random(grid,rng)
plaq = g.qcd.gauge.stencil.plaquette(U)

g.message("Initial plaquette: ", plaq)
epsilon = 0.1
Nsteps = 20
g.message("Starting zeuthen flow with fixed stepsize eps = ", epsilon, " and Nsteps = ", Nsteps)

U_flowed = g.qcd.gauge.smear.zeuthen_flow_gauge.zeuthen_flow_gauge_fixedstepsize(U,epsilon, Nsteps,0)

plaq = g.qcd.gauge.stencil.plaquette(U_flowed)

g.message(f"Plaquette after {Nsteps} steps: {plaq}")

assert(abs(plaq-1.0) < 1e-5)
g.message("Test passed!")