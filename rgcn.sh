# !/bin/sh
# ## General options
# ## -- specify queue -
# BSUB -q gpuv100
# ## -- set the job Name -
# BSUB -J rcgny
# ## -- ask for number of cores (default: 1) -
# ## -- request 6 CPU cores -
# BSUB -n 6
# ## -- specify that the cores must be on the same host -
# BSUB -R "span[hosts=1]" 
# ## -- request GPU memory (at least 14GB, it says that in the readme file on github) -
# BSUB -R "rusage[mem=16GB]"
# ## -- specify a memory limit of 16GB per slot -
# BSUB -M 16GB
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

# # Run the Python script
python rgcn.py &> output_rgcn.txt