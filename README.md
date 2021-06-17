# REMOD: a Multi-Modal Knowledge Distillation methods forDisease Relation Extraction

Authors: Yucong Lin, Keming Lu, Sheng Yu, Tianxi Cai, Marinka Zitnik

## Overview

This repository provides codes of REMOD model and training process in bi-modalities and relevant ablation study in single modality. REMOD is an approaches for multi-modal knowledge distillation ondisease relation extraction. The approach constructs a knowledge distillation framework with joint learning of knowledge graph modality and text modality. The knowledge graph consists of entity pairs of the target relations, and the texts consist of sentences related to entity pairs collected by distant supervision. The REMOD model could test on either text modality or graph modality for dealing with the missing modality problem.

## Key Idea of REMOD

Overview of REMOD model architecture is demonstrated in Figure 1. In this framework, a text encoder and a graph encoder areemployed to generate embedding of source and object entities from corpus and knowledge graph respectively.Then, a score function is used to calculate probabilities of classification with entity embedding and sharedrelation embedding. Finally, a co-training loss including cross modality knowledge distillation is adopted inorder to enhance performance of relation extraction models in both modalities.

![Figure 1. model architecture](https://github.com/Lukeming-tsinghua/REMOD/blob/master/model.pdf)

## Running the code

* **REMOD-BiModal**: code of REMOD model and training/evaluating scripts.
  + **train.sh**: the training script of REMOD. Data files in pickle format are needed. Pretrained models in both text and graph modalities can be used. The training can be started with command `bash train.sh`. An output directory will be generated and store checkpoints and results.
  + **model.py**: definitions of model structure in Pytorch
  + **main.py**: training and evaluating scripts, called in train.sh and test.sh
* **REMOD-Text**: Ablation study with text modality only. Run in the same way as REMOD-BiModal
* **REMOD-Graph**: Ablation study with graph modality only. Run in the same way as REMOD-BiModal
* **script**: python notebook files for analysis, open with Jupyter Notebook and run cells
  + **case_study.ipynb**: do case study with existing results
  + **pr_curve.ipynb**: drawing precision recall curve with existing results
  + **statistics.ipynb**: calculate the statistics of dataset, which includes data amount, lengths of sentences, etc.

## Citing

TBD

## Requirements

REMOD is tested to work under Python 3.6, packages required include

- transformers==2.8.0
- torch==1.7.0+cu110
- numpy==1.18.3
- tqdm==4.45.0
- scipy==1.6.2
- joblib==0.13.2
- dgl_cu110==0.6.1
- dgl==0.6.1
- scikit_learn==0.24.2

All packages can be installed by `pip install -r requirements.txt`

## License

REMOD is licensed under the MIT License.
