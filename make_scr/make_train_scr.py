# models=['17', '18', '19', '23', '24', '27', '28','34']
# models=['90']
runs = list(range(0,12))

block1 = ['#SBATCH --time=20:00:00',
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

model = 'simplenet'
for run in runs:
    file_name = 'run_code_run' + '_' + str(run) + '.scr'
    scr_file = open(file_name,"w+")
    scr_file.write('#!/bin/bash'+'\n')
    scr_file.write('#SBATCH --job-name=' + model + '_' + str(run) + '\n')
    for L in block1: scr_file.writelines(L+'\n')
    # scr_file.write('rm -f ../logs/train_minicohort_run' + model + '_' + str(run) + '.log' + '\n')
    # scr_file.write('rm -f ../logs/trainerror_minicohort_run'+model+'_'+str(run)+'.log'+'\n')
    scr_file.write('cd .. && python -u new_train.py --config ' + './configs/config.json'+  ' --split ' + str(run) + ' > ./logs/simplenet_full_multi/train_' + str(run)+'.log 2> ./logs/simplenet_full_multi/error_' + str(run) + '.log \n')
    scr_file.write('echo "Finished train_minicohort' + model + ' with job $SLURM_JOBID" \n')

    scr_file.close()
