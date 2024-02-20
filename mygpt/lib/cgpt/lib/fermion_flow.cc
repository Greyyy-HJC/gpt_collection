#include "lib.h"

EXPORT(Fermionflow_fixedstepsize,{
    
    PyObject* _U;
    PyObject* _chi;
    PyObject* _epsilon;
    PyObject* _Nstep;
    PyObject* _meas_interval;
    PyObject* _Nckpoints;
    PyObject* symanzik_improved;


    if (!PyArg_ParseTuple(args, "OOOOOOO", &_U, &_chi, &_epsilon, &_Nstep, &_meas_interval, &_Nckpoints, &symanzik_improved)) {
      std::cout << "Error reading arguments" << std::endl;
      return NULL;
    }

    auto grid = get_pointer<GridCartesian>(_U,"U_grid");

    LatticeGaugeFieldD U_flow(grid);
    LatticeGaugeFieldD U(grid);

    PVector<LatticeFermionD> chi_in; //using PVector is a dirty hack, only ever need one LatticeFermionD input
    LatticeFermionD chi_out(grid);
    auto grid_chi = chi_out.Grid();

    std::vector<cgpt_Lattice_base*> ferm_basis;
    cgpt_basis_fill(ferm_basis,_chi);
    cgpt_basis_fill(chi_in,ferm_basis);

    for (int mu=0;mu<Nd;mu++) {
        auto l = get_pointer<cgpt_Lattice_base>(_U,"U",mu);
        auto& Umu = compatible<iColourMatrix<vComplexD>>(l)->l;
        PokeIndex<LorentzIndex>(U,Umu,mu);
    }

   
    Real epsilon;
    int Nstep;
    int meas_interval;
    int Nckpoints;
    bool do_improved;
    cgpt_convert(_epsilon, epsilon);
    cgpt_convert(_Nstep, Nstep);
    cgpt_convert(_meas_interval, meas_interval);
    cgpt_convert(_Nckpoints,Nckpoints);
    cgpt_convert(symanzik_improved, do_improved);

    if (do_improved) {
      FermionFlow<PeriodicGimplR,ZeuthenAction<PeriodicGimplR>,LatticeFermionD> ZF(epsilon,Nstep,U,meas_interval,Nckpoints);

      if (meas_interval == 0) {
        ZF.resetActions();
      }
      ZF.smear(U_flow, chi_out, U, chi_in[0]);
    } else {
      FermionFlow<PeriodicGimplR,WilsonGaugeAction<PeriodicGimplR>,LatticeFermionD> WF(epsilon,Nstep,U,meas_interval,Nckpoints);

    
      if (meas_interval == 0) {
        WF.resetActions();
      }
      WF.smear(U_flow, chi_out, U, chi_in[0]);


    }
    // Transfrom back to stuff that gpt can deal with
    std::vector< cgpt_Lattice_base* > U_prime(4);
    for (int mu=0;mu<4;mu++) {
      auto lat = new cgpt_Lattice< iColourMatrix< vComplexD > >(grid);
      lat->l = PeekIndex<LorentzIndex>(U_flow,mu);
      U_prime[mu] = lat;
   
    }
    cgpt_Lattice_base* chi_flow;
    auto tmp = new cgpt_Lattice<iSpinColourVector<vComplexD> >(grid_chi);
    tmp->l = chi_out;
    chi_flow = tmp;


    vComplexD vScalar = 0; 
    auto U_flow_return = Py_BuildValue("(l,[i,i,i,i],s,s,[N,N,N,N])", grid, grid->_gdimensions[0],
      grid->_gdimensions[1], grid->_gdimensions[2], grid->_gdimensions[3],
      get_prec(vScalar).c_str(), "full", U_prime[0]->to_decl(), U_prime[1]->to_decl(),
      U_prime[2]->to_decl(), U_prime[3]->to_decl());

    auto chi_flow_return = Py_BuildValue("(l,[i,i,i,i],s,s,N)", grid_chi, grid_chi->_gdimensions[0],grid_chi->_gdimensions[1],grid_chi->_gdimensions[2],grid_chi->_gdimensions[3],get_prec(vScalar).c_str(), "full", chi_flow->to_decl());

    
    return Py_BuildValue("(OO)",U_flow_return,chi_flow_return);
});





EXPORT(Fermionflow_fixedstepsize_adjoint,{
    
    PyObject* _U;
    PyObject* _chi;
    PyObject* _epsilon;
    PyObject* _Nstep;
    PyObject* _meas_interval;
    PyObject* _Nckpoints;
    PyObject* symanzik_improved;


    if (!PyArg_ParseTuple(args, "OOOOOOO", &_U, &_chi, &_epsilon, &_Nstep, &_meas_interval, &_Nckpoints, &symanzik_improved)) {
      std::cout << "Error reading arguments" << std::endl;
      return NULL;
    }

    auto grid = get_pointer<GridCartesian>(_U,"U_grid");

    LatticeGaugeFieldD U_flow(grid);
    LatticeGaugeFieldD U(grid);

    PVector<LatticeFermionD> chi_in; //using PVector is a dirty hack, only ever need one LatticeFermionD input
    LatticeFermionD chi_out(grid);
    auto grid_chi = chi_out.Grid();
    LatticeFermionD chi_tmp(grid);

    std::vector<cgpt_Lattice_base*> ferm_basis;
    cgpt_basis_fill(ferm_basis,_chi);
    cgpt_basis_fill(chi_in,ferm_basis);

    for (int mu=0;mu<Nd;mu++) {
        auto l = get_pointer<cgpt_Lattice_base>(_U,"U",mu);
        auto& Umu = compatible<iColourMatrix<vComplexD>>(l)->l;
        PokeIndex<LorentzIndex>(U,Umu,mu);
    }

    
    Real epsilon;
    int Nstep;
    int meas_interval;
    int Nckpoints;
    bool do_improved;
    cgpt_convert(_epsilon, epsilon);
    cgpt_convert(_Nstep, Nstep);
    cgpt_convert(_meas_interval, meas_interval);
    cgpt_convert(_Nckpoints,Nckpoints);
    cgpt_convert(symanzik_improved, do_improved);

    if (do_improved) {
      FermionFlow<PeriodicGimplR,ZeuthenAction<PeriodicGimplR>,LatticeFermionD> ZF(epsilon,Nstep,U,meas_interval,Nckpoints);

      if (meas_interval == 0) {
        ZF.resetActions();
      }
      ZF.smear(U_flow, chi_tmp, U, chi_in[0]); 
      ZF.smear_adjoint(chi_out,chi_in[0]);

    } else {
      FermionFlow<PeriodicGimplR,WilsonGaugeAction<PeriodicGimplR>,LatticeFermionD> WF(epsilon,Nstep,U,meas_interval,Nckpoints);

      if (meas_interval == 0) {
        WF.resetActions();
      }
      WF.smear(U_flow, chi_tmp, U, chi_in[0]); 
      WF.smear_adjoint(chi_out,chi_in[0]);
    }
    // Transfrom back to stuff that gpt can deal with
    
    cgpt_Lattice_base* chi_flow;
    auto tmp = new cgpt_Lattice<iSpinColourVector<vComplexD> >(grid_chi);
    tmp->l = chi_out;
    chi_flow = tmp;

    vComplexD vScalar = 0; 
    
    auto chi_flow_return = Py_BuildValue("(l,[i,i,i,i],s,s,N)", grid_chi, grid_chi->_gdimensions[0],grid_chi->_gdimensions[1],grid_chi->_gdimensions[2],grid_chi->_gdimensions[3],get_prec(vScalar).c_str(), "full", chi_flow->to_decl());

    
    return chi_flow_return;
});


