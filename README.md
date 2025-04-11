Note: Some commands may need you to subsititute specific information into them (e.i. pid)

# Accessing Tinkercliffs on ARC Cluster via SSH

## Virtual Private Network (VPN) Requirements

Whether using OnDemand or SSH, you must use the Ivanti Secure Access Client to access ARC unless you are connected to a VT network.

You can do that by following the instructions found at: https://4help.vt.edu/sp?id=kb_article&sys_id=https:%2F%2F4help.vt.edu%2Fsp%3Fid%3Dkb_article&sysparm_article=KB0010740

## Connecting to a Login Node:

Once you are either on the VPN or a VT network connection, you can ssh into a login node like so (using TinkerCliffs as an example):
```
ssh <pid>@tinkercliffs1.arc.vt.edu
```
You will then be prompted for your Virginia Tech password, after which you will have to 2-Factor Authenticate via Duo.

This should also work on the VSCode SSH Portal.

Note: Do NOT run code on the login codes, you will have to connect to a cluster node in order to access resources.

## Setup SSH for Git:
Note: These keys might already be created on the login node.
```
ssh-keygen -t ed25519
cat <path_to_key.pub>
```
Note, this path is usually `/home/<pid>/.ssh/id_ed25519.pub`

After this, you will copy the contents of this key that are displayed by `cat` and paste them into a new SSH key in git.

# ARC Login Cluster Guide:

Here is a guide to some of the commands that I have used that I have found very helpful when using ARC. For additional commands and documentation see: https://www.docs.arc.vt.edu/

## Interactive Jobs:

Interactive jobs are designed for developing and testing code with access to the necessary resources and libraries that you will eventually run your large compute jobs on. Once your code is working properly, transitioning to a batch job is highly recommended for efficient workflow.

An interact job submission has many parameters you will need to understand:
```
interact -A capstone --partition dgx_normal_q -N 1 --ntasks-per-node=1 --gres=gpu:ampere:1 -t 3:00:00
```
Parameter Guide:
- `-A capstone`: Specifies the allocation account for your job. This must match the project you are added to (in this case, "capstone"). If you're not added to it, the job won't start.
- `--partition=dgx_normal_q`: Selects the partition (or queue). dgx_normal_q targets standard DGX compute nodes, not the developer or debug queues.
- `-N 1`: Requests 1 node. Since each DGX node has 8 GPUs with 80GB VRAM, 1 node is typically enough for most jobs unless you're doing distributed multi-node training.
- `--ntasks-per-node=1`: This sets the number of tasks (processes) per node. For most interactive GPU use, 1 task is typical, especially if you're launching a single training or inference job.
- `--gres=gpu:ampere:1`: Requests 1 Ampere GPU. The gres parameter (generic resources) lets you specify the number and type of GPU. “Ampere” usually refers to A100s on ARC, but it's always worth confirming with sinfo or documentation if unsure.

## Queue Information:
To see the queue for all compute clusters, you can do:
```
squeue
```
If you want to see the queue for a specific partition, you can do:
```
squeue -p <insert_partition_name>
```
If you want to see the jobs that are queued or running, you can do:
```
squeue --user=<pid>
```
If you want to see your allocation quotas, you can do:
```
quota
```
Finally, you can see other compute cluster information by running:
```
sinfo
```
Note: For all of these commands, you can run:
```
<command> --help
```
This will give you guidance on what you can do with each one.

## Batch Jobs (Incomplete):

When I figure out how to run batch jobs, I will place a guide here for them.

# Repository Information:
This repository consists of three major folders: `data`, `fine-tuning`, and `mech-interp`
- `data`: Contains the datasets for both fine-tuning and mechanistic interpratability analyses. Additionally includes the analysis of these datasets as well as the output data from any mechanistic interpratabiliyt methods.
- `fine-tuning`: Contains all fine-tuning materials and the links to models that we have created. The fine-tuned models themselves are hosted on public repositories on huggingface.
- `mech-interp`: This is where most of our code is. This is where all of our mechanistic interpatability materials can be found. This inlcudes everything we need to run the mechanistic interpratability jobs, bt again, the actual outputs will be stored in the data folder.

# ARC Cluster Setup:

Here we will outline a few general steps for setting up the cluster.

## Install Jupyter Extension for VSCode:

Make sure to install the necessary VSCode Jupyter extension so that you will have access to a Jupyter kernel to run your code on.

## Clone Repositories:
You will have to clone a few different repositories in different locations in order to run our code and ensure everything fits together nicely in your enviroment. Running this from your home directory will ensure that this repositories are installed in the right place.
```
git clone https://github.com/Liamayyy/team5-capstone.git &&
cd team5-capstone &&
cd fine-tuning &&
git clone https://github.com/axolotl-ai-cloud/axolotl.git
```

## Hugging Face CLI
For much of the code in this repository, you will have to create a HuggingFace account if you have not already. Once you have an account, you can request access to gated repositories and create access tokens. When creating the access token for the fine-tuning or mech-interp step make sure to have the option "Read access to contents of all public gated repos you can access" enabled. Make sure to save you token in a secure location for future use, as for each session that you use huggingface, you will have to log in to their cli. After completing these steps, you will be able to log in to HuggingFace using the following command:
```
huggingface-cli login
```
It will prompt you for your access token, and after entering it, you should be able to use axolotl to access gated models like gemma-2-2b.

Note: When prompted if you want to add the token as a git credential, I select no, as this may complicate things later down the line.

## Using Jupyter on the Compute Cluster (Not Finished Yet):
I personally like running the code on a normal python file to avoid uncessary complications, however, if you would like to use a notebook, you can with the following instructions:

First, make sure to install the necessary VSCode Jupyter extension so that you will have access to a Jupyter kernel to run your code on.

You should be able to run a Jupyter Notebook on our newly created Conda enviroment, but if you cannot select it, please complete the following steps:
1. Open the Command Palette (Ctrl+Shift+P or Cmd+Shift+P on macOS).​
2. Type and select Python: Select Interpreter.​
3. Choose Fine-Tuning Enviroment (fine-tuning) from the list. 
4. If it does not appear on the list, click on Enter interpreter path, select Find, and navigate to the `~/.conda/envs/fine-tuning/bin/python`,  your Conda environment.

Then, run these commands (in the background):
```
jupyter notebook --no-browser --port=8899
ssh -N -L 8888:localhost:8888 <pid>@tinkercliffs2.arc.vt.edu &
```
```
kill %2 && kill %1
```

## Hard Reset:
To reset everything you have done on ARC (for the most part), you can run the following commands:
```
conda env remove --name fine-tuning
rm -rf .cache/ .local/ team5-capstone/ .ipython/ .dotnet/ .conda/ .jupyter/ .lesshst/ .triton/ .nv/
```

Reminder: to eventually remove any conda enviroment, you can do the following:

```
conda env remove --name fine-tuning (or another enviroment name)
```