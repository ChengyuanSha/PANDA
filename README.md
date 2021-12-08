# PANDA: Prioritization of autism‐genes using network‐based deep‐learning approach

This repository is a re-implementation of [PANDA: Prioritization of autism‐genes using network‐based deep‐learning approach](https://onlinelibrary.wiley.com/doi/full/10.1002/gepi.22282) in PyTorch. 

## Requirements

Code environment is managed via [Anaconda](https://www.anaconda.com/) in this project.

To create an environment and install all dependencies:
```setup
conda env create -f environment.yml
```

To clone and run this project locally:

```cmd
git clone https://github.com/ChengyuanSha/PANDA
```

## File Structure

* `data` folder:
  * HMIN_edgelist.csv: Human molecular interaction network (HMIN) in an edge list format
  * labeled_genes.csv: Graph nodes labels corresponding to HMIN
* `src` folder contains implementation codes:
  * `tests` folder: tests for quality assurance.
  * `experiments` folder: other testing experiments 
  * GCN.py: definition of graph convolution network model
  * main.ipynb: the training and evaluation jupyter-notebook.
  * read_data.py: data preprocessing


## Training

Our model takes a small amount of time to train since the dataset is small.
To train my model, run ```main.ipynb``` in ```src```. The training code is under the `Model training` section.
To save model, uncomment code in `Save model` section.

## Evaluation

To evaluate my model, run ```main.ipynb``` in ```src```. 
The evaluation code is under the `Model Evaluation` section.

## Pre-trained Models

Our model trained on HMIN with labeled autism genes dataset.
You can download pretrained model in: `src/pretrained_model.pth`.


## Results
Our model (PANADA) achieves the following performance comparing with Random Forest (RF), support vector machine (SVM),
Linear Genetic Programming(LGP) on HMIN dataset:


| Model name | Accuracy | Precision | Recall | F1 score |
|--------|----------|-----------|--------|----------|
| PANADA | 86%      | 1.00      | 0.86   | 0.93     |
| RF     | 86%      | 0.98      | 0.84   | 0.91     |
| SVM    | 85%      | 0.99      | 0.85   | 0.92     |
| LGP    | 86%      | 0.98      | 0.84   | 0.91     |


## Reference 

Zhang, Y., Chen, Y., & Hu, T. (2020). 
[PANDA: Prioritization of autism‐genes using network‐based deep‐learning approach](https://onlinelibrary.wiley.com/doi/full/10.1002/gepi.22282). 
Genetic epidemiology, 44(4), 382-394.


