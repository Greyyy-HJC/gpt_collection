#if defined(GRID_CUDA)

#include <cub/cub.cuh>
#define gpucub cub
#define gpuMalloc cudaMalloc
#define gpuMemcpy cudaMemcpy
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuMemcpyToSymbol cudaMemcpyToSymbol
#define gpuError_t cudaError_t
#define gpuSuccess cudaSuccess
#define gpuStream_t cudaStream_t
#elif defined(GRID_HIP)

#include <hipcub/hipcub.hpp>
#define gpucub hipcub
#define gpuMalloc hipMalloc
#define gpuMemcpy hipMemcpy
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuMemcpyToSymbol hipMemcpyToSymbol
#define gpuError_t hipError_t
#define gpuSuccess hipSuccess
#define gpuStream_t hipStream_t
#endif

std::vector<Gamma::Algebra> Gmu4 ( {
  Gamma::Algebra::GammaX,
  Gamma::Algebra::GammaY,
  Gamma::Algebra::GammaZ,
  Gamma::Algebra::GammaT });

std::vector<Gamma::Algebra> Gmu16 ( {
  Gamma::Algebra::Gamma5,
  Gamma::Algebra::GammaT,
  Gamma::Algebra::GammaTGamma5,
  Gamma::Algebra::GammaX,
  Gamma::Algebra::GammaXGamma5,
  Gamma::Algebra::GammaY,
  Gamma::Algebra::GammaYGamma5,
  Gamma::Algebra::GammaZ,
  Gamma::Algebra::GammaZGamma5,
  Gamma::Algebra::Identity,
  Gamma::Algebra::SigmaXT,
  Gamma::Algebra::SigmaXY,
  Gamma::Algebra::SigmaXZ,
  Gamma::Algebra::SigmaYT,
  Gamma::Algebra::SigmaYZ,
  Gamma::Algebra::SigmaZT
});

#if defined(GRID_CUDA)||defined(GRID_HIP)
// extern gpuStream_t computeStream;
template<class vobj>
inline void cgpt_slice_trace_DA_sum_GPU(const PVector<Lattice<vobj>> &Data,
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

  Vector<vTComplexD> lvSum(rd * Nbasis * Nmom * Ngamma);         // will locally sum vectors first
  Vector<TComplexD> lsSum(ld * Nbasis * Nmom * Ngamma, Zero()); // sum across these down to scalars
  result.resize(fd * Nbasis * Ngamma * Nmom);              // And then global sum to return the same vector to every node
  
  int      e1 = grid->_slice_nblock[orthogdim];
  int      e2 = grid->_slice_block [orthogdim];
  int  stride = grid->_slice_stride[orthogdim];
  int ostride = grid->_ostride[orthogdim];

  commVector<vTComplexD> reduction_buffer(e1*e2);
  ExtractBuffer<TComplexD> extract_buffer(Nsimd);

  Lattice<vTComplexD> tmp(grid);
  Lattice<vobj> tmp_vobj(grid);

  VECTOR_VIEW_OPEN(mom, mom_v, AcceleratorRead);
  autoView(tmp_v, tmp, AcceleratorRead);
  auto reduction_buffer_ptr = &reduction_buffer[0];

  void *helperArray = NULL;
  vTComplexD *d_out;
  size_t temp_storage_bytes = 0;
  size_t size = e1*e2;
  
  gpuMalloc(&d_out,Nmom*Nbasis*Ngamma*rd*sizeof(vTComplexD));
  gpuError_t gpuErr = gpucub::DeviceReduce::Sum(helperArray, temp_storage_bytes, reduction_buffer_ptr, d_out, size, computeStream);
  if (gpuErr != gpuSuccess) {
    std::cout << "Encountered error during cub::deviceReduce::sum(1)! Error: " << gpuErr << std::endl;
  }

  gpuErr = gpuMalloc(&helperArray, temp_storage_bytes);
  if (gpuErr != gpuSuccess) {
    std::cout << "Encountered error during gpuMalloc (cub setup)! Error: " << gpuErr << std::endl;
  }
  
  for (int nbasis = 0; nbasis < Nbasis; nbasis++) {

    //This is the expensive call; multiplying two spincolor matrices so we take them to the outermost loop.
    //We use Grid's expression template engine instead of custom loops
    tmp_vobj = Data2[0]*Data[nbasis];    

    for (int ngamma  = 0; ngamma < Ngamma; ngamma++) {

      //use expression template engine again to multiply gamma matrices and trace down to complex numbers
      tmp = trace(tmp_vobj*Gamma(Gmu16[ngamma]));

      // now we loop over momenta and reduced orthogonal dimension rd
      // we use non-blocking accelerator_for to avoid extra synchronizations (HIP API can be slow)
      // non-blocking is fine because they all queue into computeStream and execute serially
      for (int nmom = 0; nmom < Nmom; nmom++) {
   
        for (int r = 0; r < rd; r++) {
            accelerator_forNB(s,e1 * e2, grid->Nsimd(), {
            
                int ne1 = s / e2;
                int ne2 = s % e2;

                int so = r * ostride;
                int ss = so + ne1 * stride + ne2;
                auto elem = coalescedRead(tmp_v[ss])*coalescedRead(mom_v[nmom][ss]);
                coalescedWrite(reduction_buffer_ptr[s], elem);
            });

          //reduce using cub/hipcub library, queueing into computeStream to avoid having to sync
          gpuErr = gpucub::DeviceReduce::Sum(helperArray, temp_storage_bytes, reduction_buffer_ptr, &d_out[r+rd*ngamma+rd*Ngamma*nmom+rd*Ngamma*Nmom*nbasis], size,computeStream);
          if (gpuErr != gpuSuccess) {
            std::cout << "Encountered error during cub::DeviceReduce::Sum(2)! Error: " << gpuErr << std::endl;
          }        

        }
      }
    }

  }
    
  accelerator_barrier();
  gpuMemcpy(&lvSum[0],d_out,Nmom*Nbasis*Ngamma*rd*sizeof(vTComplexD), gpuMemcpyDeviceToHost);

  VECTOR_VIEW_CLOSE(mom_v);

  thread_for(n_base, Nbasis*Nmom*Ngamma, {
    // Sum across simd lanes in the plane, breaking out orthog dir.
    ExtractBuffer<TComplexD> extracted(Nsimd); // splitting the SIMD
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
        
          result[n_base * fd + t] = lsSum[n_base * ld + lt];
        //result[n_base * fd + t] = TensorRemove(trace(lsSum[n_base * ld + lt]));

      } else {
        result[n_base * fd + t] = scalar_type(0.0); //Zero();
        //result[n_base * fd +t] = scalar_type(0.0);
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
#endif