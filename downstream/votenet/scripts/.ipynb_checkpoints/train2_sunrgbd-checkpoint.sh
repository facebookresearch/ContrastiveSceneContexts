#! /bin/bash
#################################################################################
#     File Name           :     sbatch_train.sh
#     Description         :     Run this script with 'sbatch sbatch_train.sh'
#################################################################################
#SBATCH --output=/checkpoint/jgu/jobs_output/slurm-%A_%a.out
#SBATCH --error=/checkpoint/jgu/jobs_output/slurm-%A_%a.err
#SBATCH --partition=priority
#SBATCH --comment="<ECCV deadline>"
#SBATCH --nodes=1
#SBATCH --array=0-0%1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=100g
#SBATCH --constraint=volta32gb
#SBATCH --cpus-per-task=12
#SBATCH --signal=B:USR1@60 #Signal is sent to batch script itself
#SBATCH --open-mode=append
#SBATCH --time=4320


# The ENV below are only used in distributed training with env:// initialization
# export MASTER_ADDR=${SLURM_NODELIST:0:9}${SLURM_NODELIST:10:3}
# export MASTER_PORT=29500

trap_handler () {
   echo "Caught signal: " $1
   # SIGTERM must be bypassed
   if [ "$1" = "TERM" ]; then
       echo "bypass sigterm"
   else
     # Submit a new job to the queue
     echo "Requeuing " $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID
     # SLURM_JOB_ID is a unique representation of the job, equivalent
     # to above
     scontrol requeue $SLURM_JOB_ID
   fi
}


# Install signal handler
trap 'trap_handler USR1' USR1
trap 'trap_handler TERM' TERM

# LOGDIR=/checkpoint/jgu/space/3d_ssl2/outputs/votenet/sparseconv_ssl_bs64
# MODEL=/checkpoint/s9xie/fcgf_logs/FCGF_-default/2019-11-18_18-14-50/checkpoint.pth

LOGDIR=$1
MODEL=$2

mkdir -p $LOGDIR

# main script
# CUDA_VISIBLE_DEVICES=4 \
python -u train.py \
  --dataset sunrgbd \
  --log_dir $LOGDIR \
  --num_workers 8 \
  --batch_size 64 \
  --no_height \
  --learning_rate 0.001 \
  --load_backbone $MODEL \
  --voxelization --voxel_size 0.025 --backbone 'sparseconv' | tee $LOGDIR/log.txt
