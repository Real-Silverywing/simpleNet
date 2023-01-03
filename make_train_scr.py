import os

from utils.process_config import process_config
# models=['17', '18', '19', '23', '24', '27', '28','34']
# models=['90']

config_path = '/projects/colonoscopy/group_scratch/xzhou86/simpleNet/configs/config_18wl_newimg.json'


config = process_config(config_path)
model = config.model_name
config_name = config_path.split('/')[-1]
exp_name = config.exp_name
block1 = ['#SBATCH --time=72:00:00',
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
if not os.path.exists('/projects/colonoscopy/group_scratch/xzhou86/simpleNet/logs/{}'.format(exp_name)):
    os.makedirs('/projects/colonoscopy/group_scratch/xzhou86/simpleNet/logs/{}'.format(exp_name))

runs = list(range(0,12))
os.chdir('/projects/colonoscopy/group_scratch/xzhou86/simpleNet/make_scr')



for run in runs:
    file_name = 'run_train_run' + '_' + str(run) + '.scr'
    scr_file = open(file_name,"w+")
    scr_file.write('#!/bin/bash'+'\n')
    scr_file.write('#SBATCH --job-name=' + str(run) + '_' + model  + '\n')
    for L in block1: scr_file.writelines(L+'\n')
    # scr_file.write('rm -f ../logs/train_minicohort_run' + model + '_' + str(run) + '.log' + '\n')
    # scr_file.write('rm -f ../logs/trainerror_minicohort_run'+model+'_'+str(run)+'.log'+'\n')
    scr_file.write('cd .. && python -u new_train.py --config ' + './configs/' + config_name +  ' --split ' + str(run) + ' > ./logs/' + exp_name + '/train_' + str(run)+'.log 2> ./logs/' + exp_name + '/error_' + str(run) + '.log \n')
    # scr_file.write('cd .. && python -u new_train.py --config ' + './configs/config.json'+  ' --split ' + str(run) + ' > ./logs/simplenet_full_multi/train_' + str(run)+'.log 2> ./logs/simplenet_full_multi/error_' + str(run) + '.log \n')
    scr_file.write('echo "Finished training' + model + ' with job $SLURM_JOBID" \n')

    scr_file.close()
print('.scr created')
