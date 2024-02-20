#!/bin/bash

rm -f ./sbatches/conf.dat ./sbatches/to_measure.dat
find /scratch/bbuu/qshi1/data/l64c64a076_c2pt_proton_sl4/c2pt/ -mindepth 1 -type d | sort -V >> ./sbatches/conf.dat

trans_pmax="2.2.2.0" #run from -px to px, ... -pt to pt

ArraySize=100
MaxRunningJobs=50

Nconfs=`wc -l ./sbatches/conf.dat | awk '{print $1}'`
echo number of confs is $Nconfs

for (( i=1;i<=$Nconfs;i++ ))
do
    current_conf=`awk 'NR=='$i'{print}' ./sbatches/conf.dat`
    conf_nr=`echo $current_conf | awk -F '/' '{print $NF}'`
    if [ -f /scratch/bbuu/shu1/data/Tmunu/Tmunu_conf$conf_nr.hdf5 ] && [[ `tail -n 1 /scratch/bbuu/shu1/data/Tmunu/Tmunu_conf$conf_nr_t2E.dat | awk '{print $1}'` = "4.999999999999990e+00" ]]
    then
        echo $current_conf has been measured
    else
        echo $current_conf has not been measured
        echo $current_conf >> ./sbatches/to_measure.dat
fi
done


Nconfs_to_measure=`wc -l ./sbatches/to_measure.dat | awk '{print $1}'`
if [ $(($Nconfs_to_measure%$ArraySize)) == "0" ]
then
    Narrays=$(($Nconfs_to_measure/$ArraySize))
else
    Narrays=$(($Nconfs_to_measure/$ArraySize+1))
fi

rm -f ./sbatches/sbatch_Tmunu_*.sh
this_line=1
for (( i=1;i<=$Narrays;i++ ))
do
    if (( $Nconfs_to_measure > $ArraySize ))
    then
        this_arraySize=$ArraySize
    else
            this_arraySize=$Nconfs_to_measure
    fi

echo "#!/bin/bash	
#SBATCH --job-name=Tmunu
#SBATCH --output=/u/shu1/Tmunu/Log/Tmunu_%x_%A_%a.out
#SBATCH --partition=gpuA40x4
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --constraint=\"scratch\"
#SBATCH --gpus-per-task=1
#SBATCH --account=bbuu-delta-gpu
#SBATCH --exclusive
#SBATCH --array=1-$this_arraySize%$MaxRunningJobs
#SBATCH --no-requeue
#SBATCH -t 18:00:00

module load fftw
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/sw/spack/delta-2022-03/apps/fftw/3.3.10-gcc-11.2.0-ipxfmko/lib
export C_INCLUDE_PATH=$C_INCLUDE_PATH:/sw/spack/delta-2022-03/apps/fftw/3.3.10-gcc-11.2.0-ipxfmko/include
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:/sw/spack/delta-2022-03/apps/fftw/3.3.10-gcc-11.2.0-ipxfmko/include

source /u/shu1/gpt_denn/lib/cgpt/build/source.sh
source ~/.bashrc
date
InputFiles=(" >> ./sbatches/sbatch_Tmunu_$i""_transp$trans_pmax.sh 
    for (( j=1;j<=$this_arraySize;j++ ))
    do
        current_conf=`awk 'NR=='$this_line'{print}' ./sbatches/to_measure.dat`
        conf_nr=`echo $current_conf | awk -F '/' '{print $NF}'`
	echo \""--PathConf \"/scratch/bbuu/shu1/conf_nersc/l6464f21b7130m00119m0322a_nersc.$conf_nr\" --confnum $conf_nr"\" >> ./sbatches/sbatch_Tmunu_$i""_transp$trans_pmax.sh 
        this_line=$(($this_line+1))
    done
    echo -e ")\nsrun -N 1 -n 1 python3.9 -u /u/shu1/Tmunu/get_Tmunu.py --mpi 1.1.1.1 --mpi_split 1.1.1.1 --trans_pmax $trans_pmax --PathTmunuOutFolder \"/scratch/bbuu/shu1/data/Tmunu/\" \${InputFiles[\$((\$SLURM_ARRAY_TASK_ID-1))]}\ndate" >> ./sbatches/sbatch_Tmunu_$i""_transp$trans_pmax.sh
    #sbatch ./sbatches/sbatch_Tmunu_$i""_transp$trans_pmax.sh
    Nconfs_to_measure=$(($Nconfs_to_measure-$ArraySize))
#    exit
done       
