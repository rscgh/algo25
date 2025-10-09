
## Algonauts 2025 competition

Our competition report can be found on here: https://arxiv.org/abs/2510.06235

More information on the challenge can be found on the [offical website](https://algonautsproject.com/2025/index.html), and the leaderboard is available on [codabench](https://www.codabench.org/competitions/4313/)

* Most of the revelevant scripts can be found under [`scripts/parcellated/`](scripts/parcellated/)
* The code for the Contrastive Video-fMRI Encoder can be found under [`scripts/contrastive_model/`](scripts/contrastive_model/)
* Code for the experiments with [enhanced transcripts](https://github.com/bagga005/friends_transcripts_algonauts25) and [fine-tuning slow_r50](https://github.com/bagga005/algonauts) can be found in separate repos.

To cite this work, please use:
```
Scholz, R., Bagga, K., Ahrends, C., & Barbano, C. A. (2025). Stacked Regression using Off-the-shelf, Stimulus-tuned and Fine-tuned Neural Networks for Predicting fMRI Brain Responses to Movies (Algonauts 2025 Report). [arXiv:2510.06235](https://arxiv.org/abs/2510.06235)
```

## Repository structure

* **brainannlib** is a package containing shared functions
* **scripts** - reliaze one specific step (e.g. collecting activations of an ANN to stimuli and saving them) by making use of the common functions defined in brainannlib, usually to be run on slurm nodes


## Installation 

```sh
# clone the github directory
git clone git@github.com:rscgh/algo25.git

# create a conda environment with the necessary dependencies (~4GB)
# possibly you have to first load the module, e.g by module load conda|anaconda3|miniconda
# for more troubleshooting see the paragraph below
conda env create -f environment_conda_algoenv.yml --solver=libmamba
# once we expand to more networks, this will likely have to be updated in the yml file

# activate the environment
conda activate algoenv

# [optional:] install the python kernel so you can use the environment from within jupyterhub
python -m ipykernel install --user --name algoenv --display-name "algoenv"

# go th the brainlib folder and install it as a package
# to later enable imports like
# brainannlib.anns import load_model
# can be uninstalled later using: pip uninstall brainannlib
cd algo25 #/brainannlib
pip install -e .

# Lastly set up the needed paths to the datasets. Please change the paths to suit your local system.
# You can even use the same directory for this
export ALGONAUTS_ROOT_DIR="/scratch-scc/users/robert.scholz2/cneuromod"
# per-episode ANN activations
mkdir -p $ALGONAUTS_ROOT_DIR/ann_brain_data/activations
# cummulated+reduced activations, saved regression models and predictions
mkdir -p $ALGONAUTS_ROOT_DIR/ann_brain_data/outputs

# To avoid having to rerun it everytime you start a new shell, you can add it also to your user profile
echo 'export ALGONAUTS_ROOT_DIR="/scratch-scc/users/robert.scholz2/cneuromod"' >> ~/.profile
```


## Download of the challenge files
datalad is like git with big file support

```sh
# go to the project root folder
cd $ALGONAUTS_ROOT_DIR

# the following command clones the git repo to a local dir "algonauts_2025.competitors" (without downloading the files)
datalad install -r git@github.com:courtois-neuromod/algonauts_2025.competitors.git

# donwload all the data (with r for recusrively going in subfolders and J8 for 8 parallel jobs)
# this will download ~ 2.3G of fmri-data and 109G stimuli
datalad get -r -J8 .

# or download only subdirectories
datalad get -r -J8 fmri/*
datalad get -r -J8 stimuli/*
```


## Running 

An example pipeline (leading to the current submission results) looks like the following:
```sh
python -u scripts/collect_smollm2_activations.py
python -u scripts/reduce_smollm2_activations.py
# repeat these two steps for the other modalities [...]
python -u scripts/regress_combined.py
```


## Resource requirements

```sh
# for the combined regression model fitting (min 40gb RAM, so far uses only one CPU)
srun --time=3:00:00 --export=ALL --partition=scc-cpu --ntasks=1 --nodes=1 --cpus-per-task=2 --mem=40G --pty bash

# for collecting activations, CPUs could be less
# and possibly the memory too, as whats mostly matters is the GPU memory
srun --time=6:00:00 --export=ALL --partition=scc-gpu --gres=gpu:1 --ntasks=1 --nodes=1 --cpus-per-task=12 --mem=40G --pty bash
```
