#include "lib.h"
#include "slice_trace.h"

EXPORT(slice_trace,{
    
    PyObject* _basis;
    long dim;
    if (!PyArg_ParseTuple(args, "Ol", &_basis, &dim)) {
      return NULL;
    }
    
    std::vector<cgpt_Lattice_base*> basis;
    cgpt_basis_fill(basis,_basis);

    // PVector<Lattice<vSpinColourMatrix>> tmp(basis.size());
    // for(int i=0; i<basis.size(); i++)
    //     tmp[i] = compatible<iColourMatrix<vComplexD>>(basis[i])->l;
    PVector<Lattice<vSpinColourMatrix>> tmp;
    cgpt_basis_fill(tmp, basis);

    // return cgpt_lattice_slice_trace(tmp, (int)dim);
    // PVector<Lattice<T>> basis;
    // cgpt_basis_fill(basis, _basis);
    return cgpt_slice_trace<vSpinColourMatrix>(tmp, dim);


  });

  EXPORT(slice_traceDA,{
    
    PyObject* _lhs;
    PyObject* _rhs;
    PyObject* _mom;
    long dim;
    if (!PyArg_ParseTuple(args, "OOOl", &_lhs, &_rhs, &_mom, &dim)) {
      return NULL;
    }
    
    std::vector<cgpt_Lattice_base*> basis;
    std::vector<cgpt_Lattice_base*> basis2;
    std::vector<cgpt_Lattice_base*> basism;
    cgpt_basis_fill(basis,_lhs);
    cgpt_basis_fill(basis2,_rhs);
    cgpt_basis_fill(basism, _mom);

    // PVector<Lattice<vSpinColourMatrix>> tmp(basis.size());
    // for(int i=0; i<basis.size(); i++)
    //     tmp[i] = compatible<iColourMatrix<vComplexD>>(basis[i])->l;
    PVector<Lattice<vSpinColourMatrix>> lhs;
    PVector<Lattice<vSpinColourMatrix>> rhs;
    PVector<LatticeComplex> mom;
    cgpt_basis_fill(lhs, basis);
    cgpt_basis_fill(rhs, basis2);
    cgpt_basis_fill(mom, basism);

    return cgpt_slice_traceDA<vSpinColourMatrix>(lhs, rhs, mom, dim);


  });

  EXPORT(slice_traceQPDF,{
    
    PyObject* _lhs;
    PyObject* _rhs;
    PyObject* _mom;
    long dim;
    if (!PyArg_ParseTuple(args, "OOOl", &_lhs, &_rhs, &_mom, &dim)) {
      return NULL;
    }
    
    std::vector<cgpt_Lattice_base*> basis;
    std::vector<cgpt_Lattice_base*> basis2;
    std::vector<cgpt_Lattice_base*> basism;
    cgpt_basis_fill(basis,_lhs);
    cgpt_basis_fill(basis2,_rhs);
    cgpt_basis_fill(basism, _mom);

    // PVector<Lattice<vSpinColourMatrix>> tmp(basis.size());
    // for(int i=0; i<basis.size(); i++)
    //     tmp[i] = compatible<iColourMatrix<vComplexD>>(basis[i])->l;
    PVector<Lattice<vSpinColourMatrix>> lhs;
    PVector<Lattice<vSpinColourMatrix>> rhs;
    PVector<LatticeComplex> mom;
    cgpt_basis_fill(lhs, basis);
    cgpt_basis_fill(rhs, basis2);
    cgpt_basis_fill(mom, basism);

    return cgpt_slice_traceQPDF<vSpinColourMatrix>(lhs, rhs, mom, dim);


  });
