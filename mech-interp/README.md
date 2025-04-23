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
NOTE: If you exit a GPU session and want to pull up the conda enviroment in a new GPU session, you usually only have to run this code again (as opposed to deleting the old enviroment and creating a new one). Doing this for new jobs will save lots of development time.

## Install SAE Lens
```
pip3 install sae-lens transformer_lens torch transformers datasets tqdm psutil &&
conda install jupyter -y
```
## Single Layer Comparisons
If you want to compare a specific layer of the base and fine-tuned models, you can run the command below (make sure you are in the `single-layer-analysis` directory). Let's say we want to analyze layer 8 in this case.
```
python generate_activations.py --layer 8 \
  && python generate_sparse_codes.py --layer 8 \
  && python compare_sparse_codes.py --layer 8 \
  > layer8.log 2>&1
```

If you want to run a comparison of each layer of the model, you can run (and change before running) the following bash script:
```
bash all_layers.sh
```
NOTE: This will not delete old activations and sparse codes after running, which may result in a disk space error. To prevent this, we have made a seperate script that will automatically delete the activations and sparse codes, after logging the measurements between them.
```
bash all_layers_with_cleanup.sh
```