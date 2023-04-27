"""
This file is for submitting to the cambridge HPC CPU cluster and re-starting the job
from the most recent model once the job times out
"""

import time
import subprocess as sb
import os
import numpy as np
import datetime

### set parameters ###
Nhidden = "100" # number of hidden units
seeds = [61,62,63,64,65] # seeds to run
Lplan = "8" # planning depth
T = "50" # maximum time in units of physical actions
last_save = "1000" # number of model iterations
lrate = "1e-3" #learning rate
prefix = "" #prefix something to filename

loads, load_epochs, load_fnames = [], [], ["" for seed in seeds] #some defaults for loading previous models

def generate_submission_file(seed, load = "false", load_epoch = "0", load_fname = ""):

    load = load
    load_epoch = load_epoch
    load_fname = load_fname

    options = """"-t 20 --project=. ./walls_train.jl"""
    options += " --load "+load
    options += " --load_epoch "+load_epoch
    options += " --Nhidden "+Nhidden
    options += " --seed "+str(seed)
    options += " --Lplan "+Lplan
    options += " --T "+T
    if load_fname != "":
        options += " --load_fname "+load_fname
    options += " --lrate "+lrate
    if len(prefix) > 0: options += " --prefix "+prefix
    options += " --n_epochs "+str(int(last_save)+1)
    options += """ " """

    substring = """#!/bin/bash
#!
#! Example SLURM job script for Peta4-Skylake (Skylake CPUs, OPA)
#! Last updated: Mon 13 Nov 12:25:17 GMT 2017
#!

#!#############################################################
#!#### Modify the options in this section as appropriate ######
#!#############################################################

#! sbatch directives begin here ###############################
#! Name of the job:
#SBATCH -J metaRL
#! Which project should be charged:
#SBATCH -A T2-CS156-CPU
#! How many whole nodes should be allocated?
#SBATCH --nodes=1
#! How many (MPI) tasks will there be in total? (<= nodes*32)
#! The skylake/skylake-himem nodes have 32 CPUs (cores) each.
#SBATCH --ntasks=32
#! How much wallclock time will be required?
#SBATCH --time=36:00:00
#! What types of email messages do you wish to receive?
#SBATCH --mail-type=NONE
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue

#! For 6GB per CPU, set "-p skylake"; for 12GB per CPU, set "-p skylake-himem": 
#SBATCH -p skylake

#!SBATCH -e slurm-%j.err
#!SBATCH -o slurm-%j.out

#! sbatch directives end here (put any additional directives above this line)



#! Notes:
#! Charging is determined by core number*walltime.
#! The --ntasks value refers to the number of tasks to be launched by SLURM only. This
#! usually equates to the number of MPI tasks launched. Reduce this from nodes*32 if
#! demanded by memory requirements, or if OMP_NUM_THREADS>1.
#! Each task is allocated 1 core by default, and each core is allocated 5980MB (skylake)
#! and 12030MB (skylake-himem). If this is insufficient, also specify
#! --cpus-per-task and/or --mem (the latter specifies MB per node).

#! Number of nodes and tasks per node allocated by SLURM (do not change):
numnodes=$SLURM_JOB_NUM_NODES
numtasks=$SLURM_NTASKS
mpi_tasks_per_node=$(echo "$SLURM_TASKS_PER_NODE" | sed -e  's/^\\([0-9][0-9]*\\).*$/\\1/')
#! ############################################################
#! Modify the settings below to specify the application's environment, location 
#! and launch method:

#! Optionally modify the environment seen by the application
#! (note that SLURM reproduces the environment at submission irrespective of ~/.bashrc):
. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel7/default-peta4            # REQUIRED - loads the basic environment

#! Insert additional module load commands after this line if needed:
#!source /home/ktj21/.bash_profile
module load miniconda/3
#!conda activate ktj21

#module load openmpi

#! Full path to application executable: 
application="/rds/user/ktj21/hpc-work/julia-1.7.1/bin/julia"

#! Run options for the application:
options="""+options+"""

#! Work directory (i.e. where the job will run):
workdir="$SLURM_SUBMIT_DIR"  # The value of SLURM_SUBMIT_DIR sets workdir to the directory
                            # in which sbatch is run.


#! Are you using OpenMP (NB this is unrelated to OpenMPI)? If so increase this
#! safe value to no more than 32:
export OMP_NUM_THREADS=20

#! Number of MPI tasks to be started by the application per node and in total (do not change):
np=$[${numnodes}*${mpi_tasks_per_node}]

#! The following variables define a sensible pinning strategy for Intel MPI tasks -
#! this should be suitable for both pure MPI and hybrid MPI/OpenMP jobs:
export I_MPI_PIN_DOMAIN=omp:compact # Domains are $OMP_NUM_THREADS cores in size
export I_MPI_PIN_ORDER=compact # Adjacent domains have minimal sharing of caches/sockets
#! Notes:
#! 1. These variables influence Intel MPI only.
#! 2. Domains are non-overlapping sets of cores which map 1-1 to MPI tasks.
#! 3. I_MPI_PIN_PROCESSOR_LIST is ignored if I_MPI_PIN_DOMAIN is set.
#! 4. If MPI tasks perform better when sharing caches/sockets, try I_MPI_PIN_ORDER=compact.


#! Uncomment one choice for CMD below (add mpirun/mpiexec options if necessary):

#! Choose this for a MPI code (possibly using OpenMP) using Intel MPI.
#CMD="mpirun -ppn $mpi_tasks_per_node -np $np $application $options"

#! Choose this for a pure shared-memory OpenMP parallel program on a single node:
#! (OMP_NUM_THREADS threads will be created):
CMD="$application $options"

#! Choose this for a MPI code (possibly using OpenMP) using OpenMPI:
#CMD="mpirun -npernode $mpi_tasks_per_node -np $np $application $options"

#CMD="mpirun -ppn 1 -np 1 $application $options"

###############################################################
### You should not have to change anything below this line ####
###############################################################

cd $workdir
echo -e "Changed directory to `pwd`.\\n"

JOBID=$SLURM_JOB_ID

echo -e "JobID: $JOBID\\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"

if [ "$SLURM_JOB_NODELIST" ]; then
        #! Create a machine file:
        export NODEFILE=`generate_pbs_nodefile`
        cat $NODEFILE | uniq > machine.file.$JOBID
        echo -e "\\nNodes allocated:\\n================"
        echo `cat machine.file.$JOBID | sed -e 's/\\..*$//g'`
fi

echo -e "\\nnumtasks=$numtasks, numnodes=$numnodes, mpi_tasks_per_node=$mpi_tasks_per_node (OMP_NUM_THREADS=$OMP_NUM_THREADS)"

echo -e "\\nExecuting command:\\n==================\\n$CMD\\n"

eval $CMD
    
    """

    return substring

def create_model_name( Nhidden, T, seed, Lplan, prefix = ""):
    '''somewhat ugly solution of copying this from julia'''
    #define some useful model name
    mod_name = prefix+"N"+Nhidden+"_T"+T+"_Lplan"+Lplan+"_seed"+str(seed)
    return mod_name

def last_written_epoch(task, mod_name):
    '''find the last saved checkpoint'''
    files = os.listdir('./models/'+task+'/') #all saved files
    exts = [f[len(mod_name):] for f in files if f[:len(mod_name)] == mod_name]
    epochs = []
    for ext in exts:
        try:
            epochs.append(int(ext.split('_')[1]))
        except ValueError:
            None
    
    if len(epochs) == 0: #no files exist
        print('something went wrong and we dont have any checkpoints!')
        exit()

    max_epoch = str(int(np.sort(epochs)[-1])) #largest epoch
    return max_epoch

if __name__ == '__main__':
    finished = [False for seed in seeds]
    if len(loads) == 0:
        loads = [("false" if load_fname == "" else "true") for load_fname in load_fnames]
    if len(load_epochs) == 0:
        load_epochs = ["0" for seed in seeds]
    most_recent_load_epoch = [load_epoch for load_epoch in load_epochs]
    mod_names = [create_model_name( Nhidden, T, seed, Lplan, prefix = prefix) for seed in seeds]

    starttime = datetime.datetime.now()
    starttime = starttime.strftime("%Y_%m_%d_%H_%M_%S")

    while not all(finished):

        seeds = [seeds[i] for i in range(len(seeds)) if not finished[i]]
        finished = [finished[i] for i in range(len(seeds)) if not finished[i]]

        substrs = [generate_submission_file(seeds[i], load = loads[i], load_epoch = load_epochs[i], load_fname = load_fnames[i]) for i in range(len(seeds))]
        slurm_outputs, pids = [], []
        for i in range(len(seeds)):
            slurmname = "slurm_submit_"+starttime+'_'+str(seeds[i])
            with open(slurmname, "w") as f:
                f.write(substrs[i])

            print("starting new job: ", mod_names[i], starttime, loads[i], load_epochs[i])

            slurm_outputs.append(sb.check_output(["sbatch", slurmname]).decode('UTF-8'))
            pids.append(slurm_outputs[-1].split()[-1]) #process id

            time.sleep(1) #wait one second for job to be submitted
            squeue_out = sb.check_output(["squeue", "-u", "ktj21"]).decode('UTF-8')
            assert pids[-1] in squeue_out #check that it's been submitted
            print('submitted job:', pids[-1])

        running = [True for seed in seeds]
        tic = time.time()
        while any(running):
            running = []
            time.sleep(60*60) #wait 60 mins at a time
            squeue_out = sb.check_output(["squeue", "-u", "ktj21"]).decode('UTF-8')
            for i in range(len(seeds)):
                if pids[i] in squeue_out:
                    print('still running', (time.time() - tic)/60/60)
                    running.append(True)
                else:
                    running.append(False) #finished


        ### check file list and identify most recent checkpoint ###
        load_epochs = [last_written_epoch(task, mod_name) for mod_name in mod_names]
        load_fnames = ["" for seed in seeds] #load newest model rather than old load name

        for i in range(len(seeds)):
            if load_epochs[i] >= last_save or load_epochs[i] == most_recent_load_epoch[i]:
                #if we've reached the end or not made progress
                print('finished!', load_epochs[i], last_save, most_recent_load_epoch[i])
                finished[i] = True
                loads[i] = "false"
            else:
                loads[i] = "true" # need to load from previous state
                most_recent_load_epoch[i] = load_epochs[i]
                finished[i] = False
        
