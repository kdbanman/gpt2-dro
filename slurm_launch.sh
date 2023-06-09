#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100l:4
#SBATCH --ntasks-per-node=32
#SBATCH --exclusive
#SBATCH --mem=125G
#SBATCH --time=70:00:00
#SBATCH --account=def-nidhih

rsync -av tokenized-openwebtext $SLURM_TMPDIR/

module load StdEnv/2020  gcc/9.3.0  cuda/11.4 python/3.10  arrow/11

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install -U pip
pip install --no-index -r requirements.txt


# run from test config with accelerate launche... (should be in launch script)
# ensure runs to at least first eval step, huggingface push

# write SGD config : cvar alpha 1.0

accelerate launch gpt2-openwebtext-dro.py config-1.0.json $SLURM_TMPDIR/tokenized-openwebtext

# start run
# pay attention to tqdm predicted finish time (on karma: 44hrs to complete 66% of an originally 47 hour estimate --> 67 hrs to finish --> multiply original estimate by 1.43)
# so 1.43 it for accuracy, then 1.5 it for safety
# ask slurm for it
# dump log to slurm tmpdir?  don't worry about  model repo, those are low frequency writes.
