#!/bin/sh
#BSUB -q gpuv100
#BSUB -J DTU
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1"
#BSUB -W 24:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -u fmry@dtu.dk
#BSUB -o models/output/output_%J.out
#BSUB -e models/error/error_%J.err
#BSUB -B
#BSUB -N

#Load the following in case
#module load python/3.8
module swap cuda/12.0
module swap cudnn/v8.9.1.23-prod-cuda-12.X
module swap python3/3.10.12

python3 train_VAE.py \
    --model DTU \
    --path ../../Data/ \
    --save_step 100 \
    --num_img 0.8 \
    --device cuda \
    --workers 2 \
    --epochs 50000 \
    --batch_size 100 \
    --lr 0.0002 \
    --con_training 0 \
