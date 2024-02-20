#include "proton_util.h" 

template<typename T>
void fill_seq_src_full(const PVector<Lattice<T>> & propagator, PVector<Lattice<T>> & seq_src, const int flavor = 0) {
  GridBase *grid = propagator[0].Grid();
  const int Nd = grid->_ndimension;

  VECTOR_VIEW_OPEN(seq_src, vseq_src, AcceleratorWrite);
  VECTOR_VIEW_OPEN(propagator, vprop, AcceleratorRead);

  typedef decltype(coalescedRead(vprop[0][0])) CalcElem;

  for (int polarization=0; polarization<3; polarization++) {
    std::cout << "Starting polarization " << polarization << std::endl;
    
    accelerator_for(ss, grid->oSites(), grid->Nsimd(), {
      auto local_prop = coalescedRead(vprop[0][ss]);
      CalcElem result = local_prop - local_prop; //result NEEDS to be zero, no idea how to do it more elegantly

      ProtonSeqSrcSite(local_prop, result, polarization, flavor);
      coalescedWrite(vseq_src[polarization][ss],result);
    });
    // seq_src[polarization] = seq_src[2];
    std::cout << "I am at polarization " << polarization << std::endl;
    }
    VECTOR_VIEW_CLOSE(vseq_src);
    VECTOR_VIEW_CLOSE(vprop);
} 


template<typename T>
void fill_seq_src(const PVector<Lattice<T>> &propagator, PVector<Lattice<T>> &seq_src, const int tf, const int flavor=0)
{

    //This fills the seq. src for the proton QPDF calculation. Note, the std. argument flavor=0 denotes the isospin singlet.

    GridBase *grid = propagator[0].Grid();
    const int Nd = grid->_ndimension;
 
    // autoView(vseq_src , seq_src , AcceleratorWrite);
    // autoView( vprop          , propagator     , AcceleratorRead);

    VECTOR_VIEW_OPEN(seq_src , vseq_src , AcceleratorWrite);
    VECTOR_VIEW_OPEN(propagator , vprop , AcceleratorRead);

    typedef decltype(coalescedRead(vprop[0][0])) CalcElem;


    for (int polarization=0; polarization<3; polarization++){
      std::cout << "Starting polarization " << polarization << std::endl;
      std::cout << "grid->oSites() = " << grid->oSites() << ", grid->Nsimd() = " << grid->Nsimd() << std::endl;
      std::cout << "size of seq_src = " << seq_src.size() << ", size if propagator = " << propagator.size() << std::endl;

      accelerator_for(ss, grid->oSites(), grid->Nsimd(), {

        auto local_prop = coalescedRead(vprop[0][ss]);
        //auto result = vseq_src[0][ss];
        CalcElem result = local_prop;

        ProtonSeqSrcSite(local_prop, result, polarization, flavor);
        //Proton2ptSite(local_prop, result, polarization);
	//Proton2ptSite(coalescedRead(local_prop), result, polarization);
	coalescedWrite(vseq_src[2][ss],result);
        //vseq_src[2][ss] = result;

      });

      Lattice<T> zero(grid); zero=0.0;
      LatticeInteger     coor(grid);
      LatticeCoordinate(coor, Nd-1);

      seq_src[polarization] = where(coor==Integer(tf),seq_src[2],zero);
      std::cout << "I am at polarization " << polarization << std::endl;
    }
    VECTOR_VIEW_CLOSE(vseq_src);
    VECTOR_VIEW_CLOSE(vprop);

}

template<class vobj>
inline void sliceProton_sum(const PVector<Lattice<vobj>> prop,
                    const PVector<LatticeComplex> &mom,
                    std::vector<iSinglet<ComplexD>> &result,
                    const int orthogdim)
{

  typedef typename vobj::scalar_object sobj;
  typedef typename vobj::scalar_type scalar_type;

  GridBase *grid = prop[0].Grid();
  assert(grid!=NULL);

  const int     Nd = grid->_ndimension;
  const int  Nsimd = grid->Nsimd();
  const int Nbasis = 3; //number of different polarization channels
  const int Nmom = mom.size();

  assert(orthogdim >= 0);
  assert(orthogdim < Nd);

  int fd = grid->_fdimensions[orthogdim];
  int ld = grid->_ldimensions[orthogdim];
  int rd = grid->_rdimensions[orthogdim];

  Vector<vComplexD> lvSum(rd * Nbasis * Nmom);         // will locally sum vectors first
  Vector<ComplexD> lsSum(ld * Nbasis * Nmom, 0.0); // sum across these down to scalars
  result.resize(fd * Nbasis * Nmom);              // And then global sum to return the same vector to every node
  //result.resize(fd * Nbasis);

  int      e1 = grid->_slice_nblock[orthogdim];
  int      e2 = grid->_slice_block [orthogdim];
  int  stride = grid->_slice_stride[orthogdim];
  int ostride = grid->_ostride[orthogdim];

  // sum over reduced dimension planes, breaking out orthog dir
  // Parallel over orthog direction
  //printf("in cgpt_slice_trace_sums1, before the Vector views are opened \n");
  VECTOR_VIEW_OPEN(prop, prop_v, AcceleratorRead);
  VECTOR_VIEW_OPEN(mom, mom_v, AcceleratorRead);
  auto lvSum_p = &lvSum[0];
  typedef decltype(coalescedRead(mom_v[0][0])) CalcElem;
//  typedef vComplexD CalcElem;

  accelerator_for(r, rd * Nbasis * Nmom, grid->Nsimd(), {
    //CalcElem elem;
    //CalcElem tmp;
    vComplexD elem;

    int n_mombase = r / rd;
    int n_mom = n_mombase % Nmom;
    int n_base = n_mombase / Nmom;


    int so = (r % rd) * ostride; // base offset for start of plane
    for(int n = 0; n < e1; n++){
      for(int b = 0; b < e2; b++){
        int ss = so + n * stride + b;
        vComplexD tmp = 0.0;
        Proton2ptSite(coalescedRead(prop_v[0][ss]), tmp, n_base);
        elem += tmp*TensorRemove(coalescedRead(mom_v[n_mom][ss]));
      }
    }

    //coalescedWrite(lvSum_p[r], elem);
    lvSum_p[r] = elem;
    });

  VECTOR_VIEW_CLOSE(prop_v);
  VECTOR_VIEW_CLOSE(mom_v);

  thread_for(n_base, Nbasis*Nmom, {
    // Sum across simd lanes in the plane, breaking out orthog dir.
    ExtractBuffer<ComplexD> extracted(Nsimd); // splitting the SIMD
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
        
        result[n_base * fd  + lt] = lsSum[n_base * ld + lt];
        //result[n_base * fd + t] = TensorRemove(trace(lsSum[n_base * ld + lt]));

      } else {
        result[n_base * fd  + lt] = scalar_type(0.0); //Zero();
        //result[n_base * fd +t] = scalar_type(0.0);
      }
    }
    
  });
  scalar_type* ptr = (scalar_type *) &result[0];
  //int words = fd * sizeof(sobj) / sizeof(scalar_type) * Nbasis;
  int words = fd * Nbasis * Nmom;
  //int words = fd * Nbasis;
  grid->GlobalSumVector(ptr, words);
  //printf("######### inf cgpt_slice_trace_sum1, end\n");
}

template<typename T>
PyObject* cgpt_slice_Proton(const PVector<Lattice<T>>& lhs, 
                            const PVector<LatticeComplex> mom, int dim) {
  typedef typename Lattice<T>::vector_object vobj;
  typedef typename iSinglet<ComplexD>::scalar_object sobj;
  // typedef typename vobj::scalar_type scalar_type;

  std::vector<sobj> result;
  // std::vector<scalar_type> result;
//  printf("in cgpt_slice_trace1, before going into actual function \n");
  sliceProton_sum(lhs, mom, result, dim);

  int Nbasis = 3; //number of Polarization channels
  int Nmom = mom.size();
  int Nsobj  = result.size() / Nbasis / mom.size() ;

  PyObject* ret = PyList_New(Nbasis);
  for (size_t ii = 0; ii < Nbasis; ii++) {
    PyObject* p = PyList_New(Nmom); 
    for (size_t mm=0; mm < Nmom; mm++){
      
        PyObject* corr = PyList_New(Nsobj);

        for (size_t kk = 0; kk < Nsobj; kk++) {
          int nn = ii * Nmom  * Nsobj + mm * Nsobj +  kk;
          //PyList_SET_ITEM(corr, jj, cgpt_numpy_export(result[nn]));
          // PyList_SET_ITEM(corr, kk, PyComplex_FromDoubles(result[nn].real(),result[nn].imag()));
          PyList_SET_ITEM(corr, kk, cgpt_numpy_export(result[nn]));
        }

      PyList_SET_ITEM(p, mm, corr);
    }

    PyList_SET_ITEM(ret, ii, p);
  }

  return ret;
}
