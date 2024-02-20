#include "lib.h"


EXPORT(Zeuthen_flow_gauge_fixedstepsize,{
    
    PyObject* _args;
    PyObject* _epsilon;
    PyObject* _Nstep;
    PyObject* _meas_interval;

    if (!PyArg_ParseTuple(args, "OOOO", &_args, &_epsilon, &_Nstep, &_meas_interval)) {
      std::cout << "Error reading arguments" << std::endl;
      return NULL;
    }

    auto grid = get_pointer<GridCartesian>(_args,"U_grid");

    LatticeGaugeFieldD U_flow(grid);
    LatticeGaugeFieldD U(grid);
    for (int mu=0;mu<Nd;mu++) {
        auto l = get_pointer<cgpt_Lattice_base>(_args,"U",mu);
        auto& Umu = compatible<iColourMatrix<vComplexD>>(l)->l;
        PokeIndex<LorentzIndex>(U,Umu,mu);
    }

    // Now do zeuthen flow 
    Real epsilon;
    int Nstep;
    int meas_interval;
    cgpt_convert(_epsilon, epsilon);
    cgpt_convert(_Nstep, Nstep);
    cgpt_convert(_meas_interval, meas_interval);
    LatticeColourMatrixD xform1(grid);

    typedef typename PeriodicGimplR::GaugeField GaugeLorentz;
    ZeuthenFlow<PeriodicGimplR> ZF(epsilon, Nstep, meas_interval);
    if (meas_interval == 0) {
      ZF.resetActions();
    }
    ZF.smear(U_flow, U);

    // Transfrom back to stuff that gpt can deal with
    std::vector< cgpt_Lattice_base* > U_prime(4);
    for (int mu=0;mu<4;mu++) {
      auto lat = new cgpt_Lattice< iColourMatrix< vComplexD > >(grid);
      lat->l = PeekIndex<LorentzIndex>(U_flow,mu);
      U_prime[mu] = lat;
    }

    vComplexD vScalar = 0; // TODO: grid->to_decl()
    return Py_BuildValue("(l,[i,i,i,i],s,s,[N,N,N,N])", grid, grid->_gdimensions[0], grid->_gdimensions[1], grid->_gdimensions[2], grid->_gdimensions[3], get_prec(vScalar).c_str(), "full", U_prime[0]->to_decl(), U_prime[1]->to_decl(), U_prime[2]->to_decl(), U_prime[3]->to_decl());
});
