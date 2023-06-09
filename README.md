# Exploring Dropout: A Powerful Strategy for Overfitting Prevention in Neural Networks

This repository contains the code and results for a project focused on exploring the regularization technique of dropout in neural networks. This project aims to investigate the impact of dropout on neural network performance and compare it with other benchmark regularization methods.

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Usage](#usage)
- [Results](#results)
- [Contributers](#contributers)

## Introduction
Neural networks often suffer from overfitting, where they perform well on training data but fail to generalize to new, unseen examples. The paper titled "Dropout: A Simple Way to Prevent Neural Networks from Overfitting" introduces a technique called dropout to address this problem by randomly deactivating neurons during training, forcing the network to learn robust representations and reducing reliance on specific features. The project aims to reproduce the paper's results on the CIFAR-10 datasets and compare other benchmark methods for regularization and overfitting prevention.


## Project Structure
- `reproducing-results.ipynb`: Jupyter Notebook used for reproducing the paper's results, includes the implementations of different neural network architectures with and without dropout.
- `other benchmark methods/`: Directory of Jupyer Notebooks used for comparing benchmark regularization methods, includes the implementations of different neural network architectures with dropout and other regularization techniques.
- `results/`: Contains the results obtained from training and evaluating the models plotted for easy reading.
- `project report/`: Contains the project report in both PDF and latex formats.
- `presentation.pdf`: Slides prepared for the presentation of the project.

## Requirements
To run the code in this repository, you will need the following dependencies:
- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib

## Usage
1. Clone the repository: `git clone https://github.com/your-username/neural-network-regularization-with-dropout.git`
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Open the `main.ipynb` notebook in Jupyter Notebook or JupyterLab.
4. Follow the instructions in the notebook to train and evaluate the different models.

## Results
The `results/` directory contains the obtained results, including accuracy, loss, and other evaluation metrics, for each model and regularization technique. These results can be further analyzed and visualized using the provided notebooks or custom scripts.

## Contributors
[@Ece Akdeniz](https://github.com/ece-akdeniz) <br>
[@Ata Deniz Arslaner](https://github.com/ataarslaner)
