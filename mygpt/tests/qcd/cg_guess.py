import gpt as g 

grid = g.grid([4,4,4,8], g.double)

rng = g.random("test")


U = g.qcd.gauge.random(grid,rng)

mob = g.qcd.fermion.mobius(U, mass=0.1, M5=1.4, b=1.5, c=0.5, Ls=12, boundary_phases=[1,1,1,-1])

pc = g.qcd.fermion.preconditioner
inv = g.algorithms.inverter
cg = inv.cg({"eps": 1e-8, "maxiter": 1000})

slv_5d = inv.preconditioned(pc.eo2_ne(), cg)


mob_prop = mob.propagator(slv_5d)

src = g.mspincolor(grid)

g.create.point(src,[0,0,0,0])
print("Inverting on point source")
prop_test = g(mob_prop * src)

#repeat while keeping 5d propagator
print("Inverting on point source, saving 5d prop")
src_5d = g(mob.ImportPhysicalFermionSource * src)

mob_5d_inv = slv_5d(mob)

dst_5d = g(mob_5d_inv * src_5d)

dst = g(mob.ExportPhysicalFermionSolution * dst_5d)

print("checking correctness")
print("norm of difference: ", g.norm2(dst-prop_test)/g.norm2(prop_test))

#mobius fermion with different mass
print("creating mobius fermion action with different mass")
mob2 = g.qcd.fermion.mobius(U, mass=0.11, M5=1.4, b=1.5, c=0.5, Ls=12, boundary_phases=[1,1,1,-1])

mob2_5d_inv = slv_5d(mob2)

dst2_5d = g.copy(dst_5d) #copy 5d prop solution from above

#check that previous inverter with original mass converges immediately
#Notice different syntax for inversion: function call operator() with guess as first arg vs operator*
print("Providing previous solution as initial guess to inverter, should converge instantly")
mob_5d_inv(dst2_5d, src_5d) 

#now invert with new mass
print("Inverting with previous solution as guess for different mass fermion")
mob2_5d_inv(dst2_5d, src_5d)

dst2 = g(mob2.ExportPhysicalFermionSolution * dst2_5d)


