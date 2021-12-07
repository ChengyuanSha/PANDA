# PANDA: Prioritization of autism‐genes using network‐based deep‐learning approach

This repository is a re-implementation of [PANDA: Prioritization of autism‐genes using network‐based deep‐learning approach](https://onlinelibrary.wiley.com/doi/full/10.1002/gepi.22282). 

## Requirements

Code environment is managed via [anaconda](https://www.anaconda.com/) in this project.

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
  * `tests` folder: test performance for quality assurance.
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


| Model name    | Accuracy | Precision | Recall |
|---------------|----------|-----------|--------|
| PANADA        | 86%      |       |        |
| Random Forest |          |           |        |


## Reference 

Zhang, Y., Chen, Y., & Hu, T. (2020). 
[PANDA: Prioritization of autism‐genes using network‐based deep‐learning approach](https://onlinelibrary.wiley.com/doi/full/10.1002/gepi.22282). 
Genetic epidemiology, 44(4), 382-394.


