#include "lib.h"

EXPORT(Gauge_fix,{
    
    PyObject* _metadata;
    PyObject* _args;
    PyObject* _maxiter;
    PyObject* _prec;
    PyObject* _alpha; // Add a variable for alpha
    PyObject* _ret;

    // Adjust the PyArg_ParseTuple function to include alpha
    if (!PyArg_ParseTuple(args, "OOOO", &_args, &_maxiter, &_prec, &_alpha)) {
      std::cout << "Error reading arguments" << std::endl;
      return NULL;
    }

    auto grid = get_pointer<GridCartesian>(_args,"U_grid");

    LatticeGaugeFieldD U(grid);
    for (int mu=0;mu<Nd;mu++) {
        auto l = get_pointer<cgpt_Lattice_base>(_args,"U",mu);
        auto& Umu = compatible<iColourMatrix<vComplexD>>(l)->l;
        PokeIndex<LorentzIndex>(U,Umu,mu);
    }

    Real alpha;
    int maxiter;
    Real prec;
    cgpt_convert(_prec, prec);
    cgpt_convert(_maxiter, maxiter);
    cgpt_convert(_alpha, alpha); // Convert alpha from PyObject* to Real
    LatticeColourMatrixD xform1(grid);

    FourierAcceleratedGaugeFixer<PeriodicGimplR>::SteepestDescentGaugeFix(U,xform1,alpha,maxiter,prec,prec,false,3);

    std::vector< cgpt_Lattice_base* > U_prime(4);
    for (int mu=0;mu<4;mu++) {
      auto lat = new cgpt_Lattice< iColourMatrix< vComplexD > >(grid);
      lat->l = PeekIndex<LorentzIndex>(U,mu);
      U_prime[mu] = lat;
    }
    auto trafo = new cgpt_Lattice< iColourMatrix< vComplexD> >(grid);
    trafo->l = xform1;

    vComplexD vScalar = 0; // TODO: grid->to_decl()

    return Py_BuildValue("([(l,[i,i,i,i],s,s,[N,N,N,N])],N)", grid, grid->_gdimensions[0], grid->_gdimensions[1], 
       grid->_gdimensions[2], grid->_gdimensions[3], get_prec(vScalar).c_str(), "full", U_prime[0]->to_decl(), 
       U_prime[1]->to_decl(), U_prime[2]->to_decl(), U_prime[3]->to_decl(), trafo->to_decl());
});
