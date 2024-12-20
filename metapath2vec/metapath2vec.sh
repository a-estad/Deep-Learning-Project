# !/bin/sh
# ## General options
# ## -- specify queue -
# BSUB -q gpuv100
# ## -- set the job Name -
# BSUB -J metpath2vec
# ## -- ask for number of cores (default: 1) -
# ## -- request 4 CPU cores -
# BSUB -n 4
# ## -- specify that the cores must be on the same host -
# BSUB -R "span[hosts=1]" 
# ## -- request GPU memory  -
# BSUB -R "rusage[mem=10GB]"
# ## -- specify a memory limit of 16GB per slot -
# BSUB -M 10GB
# ## -- set walltime limit: hh:mm -
# BSUB -W 09:00 
# ## -- set the email address -# please uncomment the following line and put in your e-mail address, # if you want to receive e-mail notifications on a non-default address 
# BSUB -u s203788@student.dtu.dk 
# ## -- send notification at start -
# BSUB -B 
# ## -- send notification at completion -
# BSUB -N 
# ## -- output and error log files -
# BSUB -oo meta2vec_output_%J.out
# BSUB -eo meta2vec_error_%J.err

# # Activate the virtual environment
source /zhome/45/0/155089/deeplearning/venv/bin/activate


# Run the Python script with the --epochs argument
python metapath2vec_2getembedding.py &> output_metapath2vec_get_dat_embedding_4epochs.txt
