# REMOD: a Multi-Modal Knowledge Distillation methods forDisease Relation Extraction

Authors: Yucong Lin, Keming Lu, Sheng Yu, Tianxi Cai, Marinka Zitnik

## Overview

This repository provides codes of REMOD model and training process in bi-modalities and relevant ablation study in single modality. REMOD is an approaches for multi-modal knowledge distillation ondisease relation extraction. The approach constructs a knowledge distillation framework with joint learning of knowledge graph modality and text modality. The knowledge graph consists of entity pairs of the target relations, and the texts consist of sentences related to entity pairs collected by distant supervision. The REMOD model could test on either text modality or graph modality for dealing with the missing modalityproblem.

## Key Idea of REMOD

Overview of REMOD model architecture. In this framework, a text encoder and a graph encoder areemployed to generate embedding of source and object entities from corpus and knowledge graph respectively.Then, a score function is used to calculate probabilities of classification with entity embedding and sharedrelation embedding. Finally, a co-training loss including cross modality knowledge distillation is adopted inorder to enhance performance of relation extraction models in both modalities.

## Running the code


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
