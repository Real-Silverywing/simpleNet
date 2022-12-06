
#models=['13','17','18','19', '23', '24','28','34']
models=['90']
runs = list(range(0,20))

block1 = ['#SBATCH --time=4:00:00',
    '#SBATCH --partition=gpuq',
    '#SBATCH --gres=gpu:1',
    '#SBATCH --nodes=1',
    '#SBATCH --ntasks-per-node=1',
    '#SBATCH --cpus-per-task=6',
    '#SBATCH --mail-type=ALL',
    '#SBATCH --mail-user=bwan2@jhu.edu',

    '#### load and unload modules you may need',
    '# module unload openmpi/intel',
    '# module load mvapich2/gcc/64/2.0b',
    'module unload python',
    'module load anaconda',
    'module load cuda10.2/toolkit/10.2.89',
    'eval "$(conda shell.bash hook)"',
    'conda activate /projects/skillvba/code/bwan2/dl3',

    '#### execute code and write output file to OUT-24log.',
    '# time mpiexec ./code-mvapich.x > OUT-24log']

for model in models:
    file_name = 'run_test_run' + model + '.scr'
    scr_file = open(file_name,"w+")
    scr_file.write('#!/bin/bash'+'\n')
    scr_file.write('#SBATCH --job-name=test_minicohort_run' + model + '\n')
    for L in block1: scr_file.writelines(L+'\n')
    scr_file.write('cd .. \n')
    for run in runs:
        scr_file.write('rm -f ./logs/test_minicohort_run' + model + '_' + str(run) + '.log' + '\n')
        scr_file.write('rm -f ./logs/testerror_minicohort_run'+model+'_'+str(run)+'.log'+'\n')
        scr_file.write('python -u new_test.py --config ' + './configs/config_run' + str(model) + '.json'+  ' --split ' + str(run) + ' > ./logs/test_minicohort_run' + model + '_' + str(run)+'.log 2> ./logs/testerror_minicohort_run' + model + '_' + str(run) + '.log \n')

    scr_file.write('echo "Finished test_minicohort' + model + ' with job $SLURM_JOBID" \n')
    scr_file.close()
