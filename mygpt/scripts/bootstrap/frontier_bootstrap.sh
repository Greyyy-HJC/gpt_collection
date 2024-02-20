#!/bin/bash
#
# Check debian packages
#
function check_package {
	dpkg -s $1 1> /dev/null 2> /dev/null
	if [[ "$?" != "0" ]];
	then
		echo "Package $1 needs to be installed first"
		exit 1
	fi
}

module load PrgEnv-gnu craype-accel-amd-gfx90a amd-mixed rocm cray-python cray-mpich craype-x86-trento cray-fftw
export MPICH_GPU_SUPPORT_ENABLED=1
#check_package gcc
#check_package python3
#check_package python3-pip
#check_package wget
#check_package autoconf
#check_package libssl-dev
#check_package zlib1g-dev
#check_package libfftw3-dev

#
# Install python3 if it is not yet there
#
echo "Checking numpy"
hasNumpy=$(python3 -c "import numpy" 2>&1 | grep -c ModuleNotFound)
if [[ "$hasNumpy" == "1" ]];
then
    echo "Install numpy"
    python3 -m pip install --user numpy
fi

#
# Get root directory
#
root="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." >/dev/null 2>&1 && pwd )"

#
# Precompile python
#
echo "Compile gpt"
python3 -m compileall ${root}/lib/gpt


#
# Create dependencies and download
#
dep=${root}/dependencies
if [ ! -f ${dep}/Grid/build/Grid/libGrid.a ];
then

	if [ -d ${dep} ];
	then
	    echo "$dep already exists ; rm -rf $dep before bootstrapping again"
	    exit 1
	fi

	mkdir -p ${dep}
	cd ${dep}

	#
	# Lime
	#
	wget https://github.com/usqcd-software/c-lime/tarball/master
	tar xzf master
	mv usqcd-software-c-lime* lime
	rm -f master
	cd lime
	./autogen.sh
	./configure
	make
	cd ..

	#
	# Grid
	#
	
	git clone https://github.com/dbollweg/Grid.git
	cd Grid
	git checkout gpt_proton
	./bootstrap.sh
	mkdir build
	cd build
	../configure --enable-comms=mpi-auto \
        --enable-unified=no \
        --enable-shm=nvlink \
        --enable-accelerator=hip \
	--enable-tracing=timer \
        --enable-gen-simd-width=64 \
        --enable-simd=GPU \
        --disable-fermion-reps \
        --disable-gparity \
        --with-lime=${dep}/lime \
	--with-fftw=$FFTW_ROOT \
        CXX=hipcc MPICXX=mpicxx \
        CXXFLAGS="-fPIC -I{$ROCM_PATH}/include/ -std=c++14 -I${MPICH_DIR}/include" \
        LDFLAGS="-L{$ROCM_PATH}/lib -lamdhip64 -L${MPICH_DIR}/lib -lmpi -L${CRAY_MPICH_ROOTDIR}/gtl/lib -lmpi_gtl_hsa -fopenmp -lamdhip64" HIPFLAGS=--amdgpu-target=gfx90a

	
#	../configure \
#    --enable-unified=no \
#    --enable-accelerator=hip \
#    --enable-alloc-align=4k \
#    --enable-accelerator-cshift \
#    --enable-shm=nvlink \
#    --enable-gparity=no \
#    --enable-comms=mpi-auto \
#    --disable-fermion-reps \
#    --with-lime=--with-lime=${dep}/lime \
#    --enable-simd=GPU \
#    MPICXX=mpicxx \
#    CXX=hipcc \
#    CXXFLAGS="-I/opt/rocm-4.5.0/include -std=c++14 -I${MPICH_DIR}/include -fPIC" \
#    LDFLAGS="-L${MPICH_DIR}/lib -lmpi -L${CRAY_MPICH_ROOTDIR}/gtl/lib -lmpi_gtl_hsa" \
#    HIPFLAGS=--amdgpu-target=gfx90a 
    
	cd Grid
	make -j 6
fi

if [ ! -f ${root}/lib/cgpt/build/cgpt.so ];
then
	#
	# cgpt
	#
	cd ${root}/lib/cgpt
	./make ${root}/dependencies/Grid/build 8
fi

#cd ${root}/tests
#source ${root}/lib/cgpt/build/source.sh
#./run "" "--mpi_split 1.1.1.1"

echo "To use:"
echo "source ${root}/lib/cgpt/build/source.sh"

