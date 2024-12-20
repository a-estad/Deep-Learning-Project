# !/bin/sh
# ## General options
# ## -- specify queue -
# BSUB -q gpuv100
# ## -- set the job Name -
# BSUB -J rcgn_metapath
# ## -- ask for number of cores (default: 1) -
# ## -- request 6 CPU cores -
# BSUB -n 6
# ## -- specify that the cores must be on the same host -
# BSUB -R "span[hosts=1]" 
# ## -- request GPU memory (at least 14GB, it says that in the readme file on github) -
# BSUB -R "rusage[mem=17GB]"
# ## -- specify a memory limit of 16GB per slot -
# BSUB -M 17GB
# ## -- set walltime limit: hh:mm -
# BSUB -W 02:00 
# ## -- set the email address -# please uncomment the following line and put in your e-mail address, # if you want to receive e-mail notifications on a non-default address 
# BSUB -u s203788@student.dtu.dk 
# ## -- send notification at start -
# BSUB -B 
# ## -- send notification at completion -
# BSUB -N 
# ## -- output and error log files -
# BSUB -oo rcgn_output_%J.out
# BSUB -eo rcgn_error_%J.err

# # Activate the virtual environment
source /zhome/45/0/155089/deeplearning/venv/bin/activate


# Run the Python script with the --epochs argument
#python rgcn_v2.py --epochs 50 &> output_rgcn_v2_349.txt
python /zhome/45/0/155089/deeplearning/Deep-Learning-Project/rgcn_4_wand_m2vinit.py \
--embedding_dir /zhome/45/0/155089/deeplearning/Deep-Learning-Project/embeddings \
--embedding_dim 128 \
--epochs 50 &> outputs/output_rgcn_wandb_isworkingomg_30_11_44_metpath2vecinit_64_chan_0.01_lr.txt

#python Johannestestscriptlolmetpath2vecRGCN.py --epochs 50 &> output_rgcn_wandb_metpath2vecinit_64_chan_0.01_lr_2.txt

