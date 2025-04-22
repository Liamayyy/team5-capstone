# Login Node Installation Requirements: 

These steps can be done either on the login node or the cluster node.

## Setup Fine-Tuning Miniconda Enviroment:
This will load the Miniconda3 module from ARC, create the `fine-tuning` enviroment, and add a ipykernel so that we can run Jupyter noetbooks in this conda enviroment. This is in addition to adding .local/bin to the user's PATH variable, which is necessary for accessing binaries installed by pip.
```
module reset &&
export PATH=$PATH:/home/<pid>/.local/bin &&
source ~/.bashrc &&
module load Miniconda3 &&
conda create -n fine-tuning python=3.11 &&
source activate fine-tuning &&
conda install ipykernel jupyter &&
python -m ipykernel install --user --name=fine-tuning --display-name "Fine-Tuning Enviroment (fine-tuning)"
```

Then, you will want to clone this repository and the axolotl repository if you have not already.

# Cluster Node Installation Requirements

Here, we will outline the installation requirements for our code once you have joined a compute cluster with GPU access.

## Module Reset:
Resets the modules loaded by previous users, and adds back Miniconda for our use:
```
module reset &&
export PATH=$PATH:/home/<pid>/.local/bin &&
source ~/.bashrc &&
module load Miniconda3 &&
source activate fine-tuning
```
NOTE: If you exit a GPU session and want to pull up the conda enviroment in a new GPU session, you usually only have to run this code again (as opposed to deleting the old enviroment and creating a new one). Doing this for new jobs will save lots of development time.

## Install Axolotl
Make sure you are inside the main axolotl repository or this will not work.
```
pip3 install torch &&
pip3 install -U packaging setuptools wheel &&
pip3 install --no-build-isolation -e '.[flash-attn,deepspeed]' &&
conda install jupyter -y
```
## Using Axolotl
To get started, you can run this example to make sure everything is runnning correctly.
```
axolotl train /home/<pid>/team5-capstone/fine-tuning/axolotl/examples/gemma2/qlora.yml
```
Once this is working, most of what you will have to change in order to fine-tune will be the actual .yaml file itself.

The general structure for usfine-tuning with any .yaml files is:
```
axolotl train path/to/file.yaml
```
I run the first fine-tuning example I have made with:
```
axolotl train /home/<pid>/team5-capstone/fine-tuning/medQuad_BioASQ_qlora.yml
```
For any information pertaining to the actual parameters and use of axolotl, see: https://github.com/axolotl-ai-cloud/axolotl

# Our Models:
- https://huggingface.co/Liamayyy/gemma-2-2b-medical-v2
- https://huggingface.co/Liamayyy/gemma-2-2b-medical-v1