# Dataset Description:
For this project, we combined multiple medical QA-pair datasets including:
- MedMCQA: https://github.com/medmcqa/medmcqa?tab=readme-ov-file
    - This repository provides a download link you will use to manually load the dataset
    - This accounts for ~194k QA-pairs
- MedRedQA:https://data.csiro.au/collection/csiro%3A62454
    - This is paper that will provide the download for the data
    - This accounts for ~ 45k QA-pairs
- MedQuAD: https://www.kaggle.com/datasets/pythonafroz/medquad-medical-question-answer-for-ai-research
    - Download this dataset straight from Kaggle
    - This accounts for ~15k QA-pairs
- BioASQ: https://participants-area.bioasq.org/datasets/
    - Download "Datasets for task b"  QA-pairs, from their website
    - For us, this accounted for ~5.5k, but they have been adding to this reguarly
    - We used the 13b training dataset
This collection of data allows us to properly fine-tune any model we would wish to in the domain of medical information with a diverse set of a question formats. Overall, the total number of QA-pairs comes out to ~260k. However, after combining and cleaning all the data, this comes out closer to ~236k.

Note: To access the dataset, please unzip the folder provided in the git repo:
```
unzip medical-fine-tune.zip
```
This will output a `medical-fine-tune` folder, which git will ignore. This folder only contains the final combined data.

# Dataset Combination Instructions:
If you wish to combine and clean the original data yourself, it is also provided in a subfolder. The script we used to combine these different data sources is `combine.py` and should work to recreate the dataset provided in the `medical-fine-tune` folder.

To recreate our dataset, first, `cd` into the `data` folder and then unzip `original-medical-data.zip`:
```
unzip original-medical-data.zip
```
This will output a `original-medical-data.zip` folder, which git will ignore. This folder contains the original datasets that we used, with a slightly different directory structure then the original dataset. However, the content of the data files is exactly the same.

Then, you can run the `combine.py` file by doing:
```
pip install pandas
```
And then:
```
python combine.py
```

# Dataset Analysis:
You can find our limited analysis of this dataset in the notebook: `analysis.ipynb`

TODO: Update this analysis for the new code