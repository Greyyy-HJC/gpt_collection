#include "slice_trace_gpu.h"


// sliceSum from Grid but with vector of lattices as input and traces as output
template<class vobj>
inline void cgpt_slice_trace_sums(const PVector<Lattice<vobj>> &Data,
                            //std::vector<typename vobj::scalar_object> &result,
                            std::vector<ComplexD> &result,
                            int orthogdim)
{
  ///////////////////////////////////////////////////////
  // FIXME precision promoted summation
  // may be important for correlation functions
  // But easily avoided by using double precision fields
  ///////////////////////////////////////////////////////
  typedef typename vobj::scalar_object sobj;
  typedef typename vobj::scalar_type scalar_type;

  GridBase *grid = Data[0].Grid();
  assert(grid!=NULL);

  const int     Nd = grid->_ndimension;
  const int  Nsimd = grid->Nsimd();
  const int Nbasis = Data.size();

  assert(orthogdim >= 0);
  assert(orthogdim < Nd);
  
  std::cout << GridLogMessage << "What is this?? "<<std::endl;

  int fd = grid->_fdimensions[orthogdim];
  int ld = grid->_ldimensions[orthogdim];
  int rd = grid->_rdimensions[orthogdim];
 
  std::cout << "fd = " << fd << std::endl;
  std::cout << "ld = " << ld << std::endl;
  std::cout << "rd = " << rd << std::endl;

  Vector<vobj> lvSum(rd * Nbasis);         // will locally sum vectors first
  Vector<sobj> lsSum(ld * Nbasis, Zero()); // sum across these down to scalars
  result.resize(fd * Nbasis);              // And then global sum to return the same vector to every node

  int      e1 = grid->_slice_nblock[orthogdim];
  int      e2 = grid->_slice_block [orthogdim];
  int  stride = grid->_slice_stride[orthogdim];
  int ostride = grid->_ostride[orthogdim];

  std::cout << "e1 = " << e1 << std::endl;
  std::cout << "e2 = " << e2 << std::endl;
  std::cout << "stride = " << stride << std::endl;
  std::cout << "ostride = " << ostride << std::endl;

  // sum over reduced dimension planes, breaking out orthog dir
  // Parallel over orthog direction
  VECTOR_VIEW_OPEN(Data, Data_v, AcceleratorRead);
  //open normal view for the 2nd propagator
  // autoView(Data2_v, Data2,AcceleratorRead);
  auto lvSum_p = &lvSum[0];
  typedef decltype(coalescedRead(Data_v[0][0])) CalcElem;

  accelerator_for(r, rd * Nbasis, grid->Nsimd(), {
    CalcElem elem = Zero();

    int n_base = r / rd;
    int so = (r % rd) * ostride; // base offset for start of plane
    for(int n = 0; n < e1; n++){
      for(int b = 0; b < e2; b++){
        int ss = so + n * stride + b;
        elem += coalescedRead(Data_v[n_base][ss]);
      }
    }
    coalescedWrite(lvSum_p[r], elem);
  });
  VECTOR_VIEW_CLOSE(Data_v);

  thread_for(n_base, Nbasis, {
    // Sum across simd lanes in the plane, breaking out orthog dir.
    ExtractBuffer<sobj> extracted(Nsimd); // splitting the SIMD
    Coordinate icoor(Nd);

    for(int rt = 0; rt < rd; rt++){
      extract(lvSum[n_base * rd + rt], extracted);
      for(int idx = 0; idx < Nsimd; idx++){
        grid->iCoorFromIindex(icoor, idx);
        int ldx = rt + icoor[orthogdim] * rd;
        lsSum[n_base * ld + ldx] = lsSum[n_base * ld + ldx] + extracted[idx];
      }
    }

    for(int t = 0; t < fd; t++){
      int pt = t / ld; // processor plane
      int lt = t % ld;
      if ( pt == grid->_processor_coor[orthogdim] ) {
        result[n_base * fd + t] = TensorRemove(trace(lsSum[n_base * ld + lt]));
      } else {
        result[n_base * fd + t] = scalar_type(0.0); //Zero();
      }
    }
  });
  scalar_type* ptr = (scalar_type *) &result[0];
//  int words = fd * sizeof(sobj) / sizeof(scalar_type) * Nbasis;
  int words = fd * Nbasis;
  grid->GlobalSumVector(ptr, words);
}

template<typename T>
PyObject* cgpt_slice_trace(const PVector<Lattice<T>>& basis, int dim) {
  typedef typename Lattice<T>::vector_object vobj;
  //typedef typename vobj::scalar_object sobj;
  typedef typename vobj::scalar_type scalar_type;

  //std::vector<sobj> result;
  std::vector<scalar_type> result;
  cgpt_slice_trace_sums(basis, result, dim);

  int Nbasis = basis.size();
  int Nsobj  = result.size() / basis.size();

  PyObject* ret = PyList_New(Nbasis);
  for (size_t ii = 0; ii < Nbasis; ii++) {

    PyObject* corr = PyList_New(Nsobj);
    for (size_t jj = 0; jj < Nsobj; jj++) {
      int nn = ii * Nsobj + jj;
      //PyList_SET_ITEM(corr, jj, cgpt_numpy_export(result[nn]));
      PyList_SET_ITEM(corr, jj, PyComplex_FromDoubles(result[nn].real(),result[nn].imag()));
    }

    PyList_SET_ITEM(ret, ii, corr);
  }

  return ret;
}



// sliceSum from Grid but with vector of lattices as input and traces as output, for DA
template<class vobj>
inline void cgpt_slice_trace_DA_sum(const PVector<Lattice<vobj>> &Data,
                            const PVector<Lattice<vobj>> &Data2,
                            const PVector<LatticeComplex> &mom,
                            std::vector<iSinglet<ComplexD>> &result,
                            int orthogdim)
{
  ///////////////////////////////////////////////////////
  // FIXME precision promoted summation
  // may be important for correlation functions
  // But easily avoided by using double precision fields
  ///////////////////////////////////////////////////////
  typedef typename vobj::scalar_object sobj;
  typedef typename vobj::scalar_type scalar_type;

  GridBase *grid = Data[0].Grid();
  assert(grid!=NULL);

  const int     Nd = grid->_ndimension;
  const int  Nsimd = grid->Nsimd();
  const int Nbasis = Data.size();
  const int Nmom = mom.size();

  const int Ngamma = Gmu16.size();

  assert(orthogdim >= 0);
  assert(orthogdim < Nd);

  int fd = grid->_fdimensions[orthogdim];
  int ld = grid->_ldimensions[orthogdim];
  int rd = grid->_rdimensions[orthogdim];

  Vector<vobj> lvSum(rd * Nbasis * Nmom);         // will locally sum vectors first
  Vector<sobj> lsSum(ld * Nbasis * Nmom, Zero()); // sum across these down to scalars
  result.resize(fd * Nbasis * Ngamma * Nmom);              // And then global sum to return the same vector to every node
  //result.resize(fd * Nbasis);

  int      e1 = grid->_slice_nblock[orthogdim];
  int      e2 = grid->_slice_block [orthogdim];
  int  stride = grid->_slice_stride[orthogdim];
  int ostride = grid->_ostride[orthogdim];

  // sum over reduced dimension planes, breaking out orthog dir
  // Parallel over orthog direction
  //printf("in cgpt_slice_trace_sums1, before the Vector views are opened \n");
  VECTOR_VIEW_OPEN(Data, Data_v, AcceleratorRead);
  VECTOR_VIEW_OPEN(Data2, Data2_v, AcceleratorRead);
  VECTOR_VIEW_OPEN(mom, mom_v, AcceleratorRead);
  auto lvSum_p = &lvSum[0];
  typedef decltype(coalescedRead(Data_v[0][0])) CalcElem;

  //printf("right before the accelerator for \n");
  accelerator_for(r, rd * Nbasis * Nmom, grid->Nsimd(), {
    CalcElem elem = Zero();
    CalcElem tmp;

    int n_mombase = r / rd;
    int n_mom = n_mombase % Nmom;
    int n_base = n_mombase / Nmom;


    int so = (r % rd) * ostride; // base offset for start of plane
    for(int n = 0; n < e1; n++){
      for(int b = 0; b < e2; b++){
        int ss = so + n * stride + b;
        elem += coalescedRead(Data2_v[0][ss])*coalescedRead(Data_v[n_base][ss])*coalescedRead(mom_v[n_mom][ss]);
      }
    }

    

    coalescedWrite(lvSum_p[r], elem);
  });
  VECTOR_VIEW_CLOSE(Data2_v);
  VECTOR_VIEW_CLOSE(Data_v);
  VECTOR_VIEW_CLOSE(mom_v);

  //printf("######### inf cgpt_slice_trace_sum1, before thread_for \n");
  thread_for(n_base, Nbasis*Nmom, {
    // Sum across simd lanes in the plane, breaking out orthog dir.
    ExtractBuffer<sobj> extracted(Nsimd); // splitting the SIMD
    Coordinate icoor(Nd);

    for(int rt = 0; rt < rd; rt++){
      extract(lvSum[n_base * rd + rt], extracted);
      for(int idx = 0; idx < Nsimd; idx++){
        grid->iCoorFromIindex(icoor, idx);
        int ldx = rt + icoor[orthogdim] * rd;
        lsSum[n_base * ld + ldx] = lsSum[n_base * ld + ldx] + extracted[idx];
      }
    }
    for(int mu=0;mu<Ngamma;mu++){
    for(int t = 0; t < fd; t++){
      int pt = t / ld; // processor plane
      int lt = t % ld;
      if ( pt == grid->_processor_coor[orthogdim] ) {
        
          result[n_base * Ngamma * fd + mu * fd + t] = TensorRemove(trace(lsSum[n_base * ld + lt]*Gamma(Gmu16[mu])));
        //result[n_base * fd + t] = TensorRemove(trace(lsSum[n_base * ld + lt]));

      } else {
        result[n_base * Ngamma * fd + mu * fd + t] = scalar_type(0.0); //Zero();
        //result[n_base * fd +t] = scalar_type(0.0);
      }
    }
    }
  });
  scalar_type* ptr = (scalar_type *) &result[0];
  //int words = fd * sizeof(sobj) / sizeof(scalar_type) * Nbasis;
  int words = fd * Ngamma * Nbasis * Nmom;
  //int words = fd * Nbasis;
  grid->GlobalSumVector(ptr, words);
  //printf("######### inf cgpt_slice_trace_sum1, end\n");
}

template<typename T>
PyObject* cgpt_slice_traceDA(const PVector<Lattice<T>>& lhs, const PVector<Lattice<T>>& rhs, 
                            const PVector<LatticeComplex> mom, int dim) {
  typedef typename Lattice<T>::vector_object vobj;
  typedef typename iSinglet<ComplexD>::scalar_object sobj;
  // typedef typename vobj::scalar_type scalar_type;

  std::vector<sobj> result;
  // std::vector<scalar_type> result;
//  printf("in cgpt_slice_trace1, before going into actual function \n");

  #if defined(GRID_CUDA)||defined(GRID_HIP)
  cgpt_slice_trace_DA_sum_GPU(lhs, rhs, mom, result, dim);
  #else
  cgpt_slice_trace_DA_sum(lhs,rhs,mom,result,dim);
  #endif

  int Nbasis = lhs.size();
  int Nmom = mom.size();
  int NGamma = Gmu16.size();
  int Nsobj  = result.size() / lhs.size() / mom.size() / Gmu16.size() ;

  PyObject* ret = PyList_New(Nbasis);
  for (size_t ii = 0; ii < Nbasis; ii++) {
    PyObject* p = PyList_New(Nmom); 
    for (size_t mm=0; mm < Nmom; mm++){

      PyObject* mu = PyList_New(NGamma);

      for (size_t jj = 0; jj < NGamma; jj++) {
      
        PyObject* corr = PyList_New(Nsobj);

        for (size_t kk = 0; kk < Nsobj; kk++) {
          int nn = ii * Nmom * NGamma * Nsobj + mm * NGamma * Nsobj + jj * Nsobj + kk;
          //PyList_SET_ITEM(corr, jj, cgpt_numpy_export(result[nn]));
          // PyList_SET_ITEM(corr, kk, PyComplex_FromDoubles(result[nn].real(),result[nn].imag()));
          PyList_SET_ITEM(corr, kk, cgpt_numpy_export(result[nn]));
        }

        PyList_SET_ITEM(mu, jj, corr);

      }
      PyList_SET_ITEM(p, mm, mu);
    }

    PyList_SET_ITEM(ret, ii, p);
  }

  return ret;
}


// sliceSum from Grid but with vector of lattices as input and traces as output, for DA
template<class vobj>
inline void cgpt_slice_trace_QPDF_sum(const PVector<Lattice<vobj>> &Data,
                            const PVector<Lattice<vobj>> &Data2,
                            const PVector<LatticeComplex> &mom,
                            std::vector<iSinglet<ComplexD>> &result,
                            int orthogdim)
{
  ///////////////////////////////////////////////////////
  // FIXME precision promoted summation
  // may be important for correlation functions
  // But easily avoided by using double precision fields
  ///////////////////////////////////////////////////////
  typedef typename vobj::scalar_object sobj;
  typedef typename vobj::scalar_type scalar_type;

  GridBase *grid = Data[0].Grid();
  assert(grid!=NULL);

  const int     Nd = grid->_ndimension;
  const int  Nsimd = grid->Nsimd();
  const int Nbasis = Data2.size();
  const int Nmom = mom.size();

  const int Ngamma = Gmu16.size();

  assert(orthogdim >= 0);
  assert(orthogdim < Nd);

  int fd = grid->_fdimensions[orthogdim];
  int ld = grid->_ldimensions[orthogdim];
  int rd = grid->_rdimensions[orthogdim];

  Vector<vobj> lvSum(rd * Nbasis * Nmom);         // will locally sum vectors first
  Vector<sobj> lsSum(ld * Nbasis * Nmom, Zero()); // sum across these down to scalars
  result.resize(fd * Nbasis * Ngamma * Nmom);              // And then global sum to return the same vector to every node
  //result.resize(fd * Nbasis);

  int      e1 = grid->_slice_nblock[orthogdim];
  int      e2 = grid->_slice_block [orthogdim];
  int  stride = grid->_slice_stride[orthogdim];
  int ostride = grid->_ostride[orthogdim];

  // sum over reduced dimension planes, breaking out orthog dir
  // Parallel over orthog direction
  //printf("in cgpt_slice_trace_sums1, before the Vector views are opened \n");
  VECTOR_VIEW_OPEN(Data, Data_v, AcceleratorRead);
  VECTOR_VIEW_OPEN(Data2, Data2_v, AcceleratorRead);
  VECTOR_VIEW_OPEN(mom, mom_v, AcceleratorRead);
  auto lvSum_p = &lvSum[0];
  typedef decltype(coalescedRead(Data_v[0][0])) CalcElem;

  //printf("right before the accelerator for \n");
  accelerator_for(r, rd * Nbasis * Nmom, grid->Nsimd(), {
    CalcElem elem = Zero();
    CalcElem tmp;

    int n_mombase = r / rd;
    int n_mom = n_mombase % Nmom;
    int n_base = n_mombase / Nmom;


    int so = (r % rd) * ostride; // base offset for start of plane
    for(int n = 0; n < e1; n++){
      for(int b = 0; b < e2; b++){
        int ss = so + n * stride + b;
        elem += coalescedRead(Data2_v[n_base][ss])*coalescedRead(Data_v[0][ss])*coalescedRead(mom_v[n_mom][ss]);
      }
    }

    

    coalescedWrite(lvSum_p[r], elem);
  });
  VECTOR_VIEW_CLOSE(Data2_v);
  VECTOR_VIEW_CLOSE(Data_v);
  VECTOR_VIEW_CLOSE(mom_v);

  //printf("######### inf cgpt_slice_trace_sum1, before thread_for \n");
  thread_for(n_base, Nbasis*Nmom, {
    // Sum across simd lanes in the plane, breaking out orthog dir.
    ExtractBuffer<sobj> extracted(Nsimd); // splitting the SIMD
    Coordinate icoor(Nd);

    for(int rt = 0; rt < rd; rt++){
      extract(lvSum[n_base * rd + rt], extracted);
      for(int idx = 0; idx < Nsimd; idx++){
        grid->iCoorFromIindex(icoor, idx);
        int ldx = rt + icoor[orthogdim] * rd;
        lsSum[n_base * ld + ldx] = lsSum[n_base * ld + ldx] + extracted[idx];
      }
    }
    for(int mu=0;mu<Ngamma;mu++){
    for(int t = 0; t < fd; t++){
      int pt = t / ld; // processor plane
      int lt = t % ld;
      if ( pt == grid->_processor_coor[orthogdim] ) {
        
        result[n_base * Ngamma * fd + mu * fd + t] = TensorRemove(trace(lsSum[n_base * ld + lt]*Gamma(Gmu16[mu])));
        //result[n_base * fd + t] = TensorRemove(trace(lsSum[n_base * ld + lt]));

      } else {
        result[n_base * Ngamma * fd + mu * fd + t] = scalar_type(0.0); //Zero();
        //result[n_base * fd +t] = scalar_type(0.0);
      }
    }
    }
  });
  scalar_type* ptr = (scalar_type *) &result[0];
  //int words = fd * sizeof(sobj) / sizeof(scalar_type) * Nbasis;
  int words = fd * Ngamma * Nbasis * Nmom;
  //int words = fd * Nbasis;
  grid->GlobalSumVector(ptr, words);
  //printf("######### inf cgpt_slice_trace_sum1, end\n");
}

template<typename T>
PyObject* cgpt_slice_traceQPDF(const PVector<Lattice<T>>& lhs, const PVector<Lattice<T>>& rhs, 
                            const PVector<LatticeComplex> mom, int dim) {
  typedef typename Lattice<T>::vector_object vobj;
  typedef typename iSinglet<ComplexD>::scalar_object sobj;
  // typedef typename vobj::scalar_type scalar_type;

  std::vector<sobj> result;
  // std::vector<scalar_type> result;
//  printf("in cgpt_slice_trace1, before going into actual function \n");
  cgpt_slice_trace_QPDF_sum(lhs, rhs, mom, result, dim);

  int Nbasis = rhs.size();
  int Nmom = mom.size();
  int NGamma = Gmu16.size();
  int Nsobj  = result.size() / rhs.size() / mom.size() / Gmu16.size() ;

  PyObject* ret = PyList_New(Nbasis);
  for (size_t ii = 0; ii < Nbasis; ii++) {
    PyObject* p = PyList_New(Nmom); 
    for (size_t mm=0; mm < Nmom; mm++){

      PyObject* mu = PyList_New(NGamma);

      for (size_t jj = 0; jj < NGamma; jj++) {
      
        PyObject* corr = PyList_New(Nsobj);

        for (size_t kk = 0; kk < Nsobj; kk++) {
          int nn = ii * Nmom * NGamma * Nsobj + mm * NGamma * Nsobj + jj * Nsobj + kk;
          //PyList_SET_ITEM(corr, jj, cgpt_numpy_export(result[nn]));
          // PyList_SET_ITEM(corr, kk, PyComplex_FromDoubles(result[nn].real(),result[nn].imag()));
          PyList_SET_ITEM(corr, kk, cgpt_numpy_export(result[nn]));
        }

        PyList_SET_ITEM(mu, jj, corr);

      }
      PyList_SET_ITEM(p, mm, mu);
    }

    PyList_SET_ITEM(ret, ii, p);
  }

  return ret;
}
