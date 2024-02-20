#include "lib.h"

EXPORT(ProjectSU3,{

    void* _in;
    void* _out;
    if (!PyArg_ParseTuple(args, "ll", &_in, &_out)) {
      std::cout << "C code: problems parsing arguments" << std::endl;
      return NULL;
    }

    cgpt_Lattice_base* in = (cgpt_Lattice_base*)_in;
    cgpt_Lattice_base* out = (cgpt_Lattice_base*)_out;

    auto& In = compatible<iColourMatrix<vComplexD>>(in)->l;
    auto& Out = compatible<iColourMatrix<vComplexD>>(out)->l;
    
    Projectnew(In,Out);



    return PyLong_FromLong(0);    

});

EXPORT(ProjectStout,{

    void* _in;
    if (!PyArg_ParseTuple(args, "l", &_in)) {
      std::cout << "C code: problems parsing arguments" << std::endl;
      return NULL;
    }

    cgpt_Lattice_base* in = (cgpt_Lattice_base*)_in;


    auto& In = compatible<iColourMatrix<vComplexD>>(in)->l;

    LatticeColourMatrixD tmp(In.Grid());

    tmp = Ta(In);
    
    Smear_Stout<PeriodicGimplR> stout;

    stout.exponentiate_iQ(In, tmp);



    return PyLong_FromLong(0);    

});
