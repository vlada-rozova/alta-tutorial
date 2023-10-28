# Fine-tune BERT to recognize entities in CHIFIR

Put chifir data folder and metadata file under this directory or change the path in the notebooks.

## 1. Prepare data
Refer to `data.ipynb`: convert `.txt` and `.ann` files to fit the format to be modeled by huggingface transformers.


## 2. Fine-tune BERT
Refer to `modeling.ipynb`: three BERT checkpoints are evaluated: BERT, ClinicalBERT, and PubMedBERT. 


Best test results and outputs stored in `test_results.json` and `test_predictions.txt`