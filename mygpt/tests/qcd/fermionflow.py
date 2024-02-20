import gpt as g


grid = g.grid([8,8,8,8], g.double)

rng = g.random('test')

U = g.qcd.gauge.random(grid,rng)
chi = g.vspincolor(grid)
eta = g.vspincolor(grid)

chi_flowed = g.vspincolor(grid)
g.message(type(chi_flowed))
rng.cnormal(chi)
rng.cnormal(eta)
plaq = g.qcd.gauge.stencil.plaquette(U)

g.message("Initial plaquette: ", plaq)
g.message(f"Initial chi norm: {g.norm2(chi)}")
epsilon = 0.1
Nsteps = 20
g.message("Starting fermion flow with Wilson action and fixed stepsize eps = ", epsilon, " and Nsteps = ", Nsteps)

U_flowed,chi_flowed = g.qcd.fermion.flow.Fermionflow_fixedstepsize(U,chi, epsilon, Nsteps)

plaq = g.qcd.gauge.stencil.plaquette(U_flowed)

g.message(f"Plaquette after {Nsteps} steps: {plaq}")
chi_color = g.vcolor(grid)
g.message(f"Chi_flowed norm after {Nsteps} steps: {g.norm2(chi_flowed)}")
g.message(type(chi_flowed))
assert abs(plaq-1.0) < 1e-5 
g.message("Test passed!")


g.message("Testing adjoint flow:")

eta_adjoint_flowed = g.qcd.fermion.flow.Fermionflow_fixedstepsize_adjoint(U,eta, 0.005, 400)

prod_t = g.inner_product(chi_flowed,eta)
g.message(f"<chi(t)|eta(t)> = {prod_t}")
prod_0 = g.inner_product(chi,eta_adjoint_flowed)

g.message(f"<chi(0)|eta(0)> = {prod_0}")
g.message(f"|<chi(t)|eta(t)-<chi(0)|eta(0)>|/vol = {abs(prod_t-prod_0)/pow(8,4)}")
assert abs(prod_t-prod_0)/pow(8,4) < 1e-5
g.message("Adjoint fermion flow test passed!")

g.message("Starting fermion flow with Zeuthen action and fixed stepsize eps = ", epsilon, " and Nsteps = ", Nsteps)


U_flowed,chi_flowed = g.qcd.fermion.flow.Fermionflow_fixedstepsize(U,chi, epsilon, Nsteps, improvement=True)

plaq = g.qcd.gauge.stencil.plaquette(U_flowed)

g.message(f"Plaquette after {Nsteps} steps: {plaq}")
chi_color = g.vcolor(grid)
g.message(f"Chi_flowed norm after {Nsteps} steps: {g.norm2(chi_flowed)}")
g.message(type(chi_flowed))
assert abs(plaq-1.0) < 1e-5 
g.message("Test passed!")


g.message("Testing adjoint flow:")

eta_adjoint_flowed = g.qcd.fermion.flow.Fermionflow_fixedstepsize_adjoint(U,eta, 0.005, 400, improvement=True)

prod_t = g.inner_product(chi_flowed,eta)
g.message(f"<chi(t)|eta(t)> = {prod_t}")
prod_0 = g.inner_product(chi,eta_adjoint_flowed)

g.message(f"<chi(0)|eta(0)> = {prod_0}")
g.message(f"|<chi(t)|eta(t)-<chi(0)|eta(0)>|/vol = {abs(prod_t-prod_0)/pow(8,4)}")
assert abs(prod_t-prod_0)/pow(8,4) < 1e-5
g.message("Adjoint fermion flow test passed!")