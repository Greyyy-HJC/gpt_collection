module load PrgEnv-gnu craype-accel-amd-gfx90a amd-mixed rocm cray-python cray-mpich craype-x86-trento cray-fftw
export MPICH_GPU_SUPPORT_ENABLED=1

./make /ccs/home/jinchen/gpt_gpu/dependencies/Grid/build 8