
models=['12','13','14','15']
runs = list(range(0,20))

block1 = ['#SBATCH --time=2-00:00:00',
    '#SBATCH --partition=gpuk80',
    '#SBATCH --gres=gpu:1',
    '#SBATCH --nodes=1',
    '#SBATCH --ntasks-per-node=1',
    '#SBATCH --cpus-per-task=6',
    '#SBATCH --mail-type=ALL',
    '#SBATCH --mail-user=svedula3@jhu.edu',

    '#### load and unload modules you may need',
    '# module unload openmpi/intel',
    '# module load mvapich2/gcc/64/2.0b',
    'module restore condagit',
    'conda activate dl',

    '#### execute code and write output file to OUT-24log.',
    '# time mpiexec ./code-mvapich.x > OUT-24log']

for model in models:
    for run in runs:
        file_name = 'run_test_'+model+'_'+str(run)+'.scr'
        scr_file = open(file_name,"w+")
        scr_file.write('#!/bin/bash'+'\n')
        scr_file.write('#SBATCH --job-name=testnew'+model+'\n')
        for L in block1: scr_file.writelines(L+'\n')
        scr_file.write('rm testnew'+str(run)+'.log'+'\n')
        scr_file.write('cd .. && python -u new_test.py --split '+ str(run) + ' > ./marcc_scripts/newtest'+str(run)+'.log 2> ./marcc_scripts/newtesterror'+str(run)+'.log \n')
        scr_file.write('echo "Finished testnew'+model+' with job $SLURM_JOBID" \n')

        scr_file.close()


