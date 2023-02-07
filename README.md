# Co-Driven Recognition of Chinese Semantic Entailment via the Fusion of Transformer and HowNet Sememes Knowledge

---
 This is the code of our paper for ESWC-2023, there are some explanations for it
---
1. For non-pretraining models, you need to run Pre-processing.py to generate data before running the models. For pretaining models, please download BERT models before you run hownet_bert.py.
2. We just take BERT model and BQ dataset for example, it is easy to expand to other text semantic matching datasets or replace with other pretraining models.
3. Our experimental results on the BQ, AFQMC and PAWSX-zh datasets are as follows:

- BQ corpus

| models   |    pretaining model   | Acc       |F1        |
| :------- | :-------------------- |:----------|:---------|
| DSSM            | × |     77.12       |     76.47       |
| MwAN            | × |      73.99      |    73.29        |
| DRCN            | × |      74.65      |       76.02     |
| Ours            | × |     78.81       |     76.62       |
| Improvement     | × |     +2.19%      |     +1.96%      |
| BERT-wwm-ext    | √ |      84.71      |     83.94       |
| BERT            | √ |       84.50     |      84.00      |
| ERNIE           | √ |       84.67     |      84.20      |
| Ours-BERT       | √ |      84.82      |      84.33      |
| Improvement     | √ |     +0.177%     |     +0.464%     |





- AFQMC dataset

| models   |    pretaining model   | Acc       |F1        |
| :------- | :-------------------- |:----------|:---------|
| DSSM            | × |     57.02       |     30.75       |
| MwAN            | × |      65.43      |    28.63       |
| DRCN            | × |      66.05      |       40.60     |
| Ours            | × |     66.62       |     42.93       |
| Improvement     | × |     +0.86%       |     +5.7%       |
| BERT-wwm-ext    | √ |      81.76      |     80.62       |
| BERT            | √ |       81.43     |      79.77      |
| ERNIE           | √ |       81.54     |      80.81      |
| Ours-BERT       | √ |      81.84      |      81.93      |
| Improvement     | √ |     +0.097%     |     +1.38%     |


- PAWSX-zh dataset

| models   |    pretaining model   | Acc       |F1        |
| :------- | :-------------------- |:----------|:---------|
| DSSM            | × |     42.64       |     59.43      |
| MwAN            | × |      52.70      |    52.65       |
| DRCN            | × |      61.24      |       56.52     |
| Ours            | × |     62.55       |     59.72       |
| Improvement     | × |     +2.13%       |     +0.48%       |
| BERT-wwm-ext    | √ |      77.23      |     76.52       |
| BERT            | √ |       77.06     |      77.16      |
| ERNIE           | √ |       78.02     |      77.59      |
| Ours-BERT       | √ |      78.33      |      77.96      |
| Improvement     | √ |     +0.397%     |     +0.476%     |
