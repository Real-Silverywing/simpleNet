
import os
from utils.process_config import process_config
#models=['13','17','18','19', '23', '24','28','34']
config_path = '/projects/colonoscopy/group_scratch/xzhou86/simpleNet/configs/config_cnn_resample_weight.json'


config = process_config(config_path)
model = config.model_name
config_name = config_path.split('/')[-1]
exp_name = config.exp_name
block1 = ['#SBATCH --time=4:00:00',
    '#SBATCH --partition=gpuq',
    '#SBATCH --gres=gpu:1',
    '#SBATCH --nodes=1',
    '#SBATCH --ntasks-per-node=1',
    '#SBATCH --cpus-per-task=6',


    '#### load and unload modules you may need',
    '# module unload openmpi/intel',
    '# module load mvapich2/gcc/64/2.0b',
    'module unload python',
    'module load anaconda',
    'module load cuda10.2/toolkit/10.2.89',
    'eval "$(conda shell.bash hook)"',
    'conda activate /projects/colonoscopy/code/xzhou86/torch',

    '#### execute code and write output file to OUT-24log.',
    '# time mpiexec ./code-mvapich.x > OUT-24log']

runs = list(range(0,12))
os.chdir('/projects/colonoscopy/group_scratch/xzhou86/simpleNet/make_scr')
for run in runs:
    file_name = 'run_test_run' + '_' + str(run) + '.scr'
    scr_file = open(file_name,"w+")
    scr_file.write('#!/bin/bash'+'\n')
    scr_file.write('#SBATCH --job-name=test_' + model + '\n')
    for L in block1: scr_file.writelines(L+'\n')
    # scr_file.write('rm -f ./logs/test_minicohort_run' + model + '_' + str(run) + '.log' + '\n')
    # scr_file.write('rm -f ./logs/testerror_minicohort_run'+model+'_'+str(run)+'.log'+'\n')
    scr_file.write('cd .. && python -u new_test.py --config ' + './configs/' + config_name +  ' --split ' + str(run) + ' > ./logs/' + exp_name + '/test_' + str(run)+'.log 2> ./logs/' + exp_name + '/testerror_' + str(run) + '.log \n')

    scr_file.write('echo "Finished test_' + model + ' with job $SLURM_JOBID" \n')
    scr_file.close()
