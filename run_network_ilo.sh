#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=compsci-gpu

set -ex
export CUDA_HOME=/usr/local/cuda-11.1/
export PATH=/usr/local/cuda-11.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64
alias gcc=gcc-5
alias g++=g++-5

rm -rf ~/.cache/torch_extensions

python3 run_network_ilo.py -input_dir /usr/xtmp/jl888/pulse-ae/celeba_hq_preprocessed/images_32/00021/ -output_dir ~/1105 -facebank_dir /usr/xtmp/jl888/pulse-ae/celeba_hq_preprocessed/images_ref -duplicates 1 -eps 0 -steps 5000 -loss_str 100*L2+1*ARCFACE+1*LATENTNORM+0*LPIPS+0*VGG -opt_name adamax -num_layers_arcface 1 -latent_radius 96 -use_initialization random -save_best -arcface_use_ref -learning_rate 1e-2