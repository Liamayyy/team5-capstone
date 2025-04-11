# Login Node Installation Requirements

These steps can be done either on the login node or the cluster node.

## Setup Mech-Interp Miniconda Enviroment:
First we will create a conda enviroment to run our code in:
```
module reset &&
export PATH=$PATH:/home/<pid>/.local/bin &&
source ~/.bashrc &&
module load Miniconda3 &&
conda create -n mech-interp python=3.11 &&
source activate mech-interp &&
conda install ipykernel jupyter &&
python -m ipykernel install --user --name=mech-interp --display-name "Mechanistic Interpratability Enviroment (mech-interp)"
```

# Cluster Node Installation Requirements

Here, we will outline the installation requirements for our code once you have joined a compute cluster with GPU access.

## Module Reset:
Resets the modules loaded by previous users, and adds back Miniconda for our use:
```
module reset &&
export PATH=$PATH:/home/<pid>/.local/bin &&
source ~/.bashrc &&
module load Miniconda3 &&
source activate mech-interp
```

## Install SAE Lens
Make sure you are inside the main axolotl repository or this will not work.
```
pip3 install git+https://github.com/jbloomAus/SAELens.git &&
pip3 install torch transformers &&
conda install jupyter -y
```