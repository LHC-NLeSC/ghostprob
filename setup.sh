source /cvmfs/sft.cern.ch/lcg/views/LCG_102/x86_64-centos7-gcc11-opt/setup.sh

# CUDA, sets $CUDACXX to nvcc
source /cvmfs/sft.cern.ch/lcg/contrib/cuda/11.4/x86_64-centos7/setup.sh
export CUDNN_ROOT=/project/bfys/suvayua/build/cuda
export TRT_ROOT=/project/bfys/suvayua/build/TensorRT-8.4.0.6
export UMESIMD_ROOT_DIR=/project/bfys/suvayua/build

