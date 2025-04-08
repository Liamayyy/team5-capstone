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
This collection of data allows us to properly fine-tune any model we would wish to in the domain of medical information with a diverse set of a question formats. Overall, the total number of QA-pairs comes out to ~260k. However, after combining and cleaning all the data, this comes out closer to ____.

# Dataset Combination Instructions:
```
pip install pandas
```
