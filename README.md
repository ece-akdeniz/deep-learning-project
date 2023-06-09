# Exploring Dropout: A Powerful Strategy for Overfitting Prevention in Neural Networks

This repository contains the code and results for a project focused on exploring the regularization technique of dropout in neural networks. The project aims to reproduce the findings of a seminal paper on dropout and extend the analysis to compare it with other benchmark regularization methods.

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Usage](#usage)
- [Results](#results)
- [Contributers](#contributers)

## Introduction
Neural networks often suffer from overfitting, where they perform well on training data but fail to generalize to new, unseen examples. Dropout is a regularization technique that addresses overfitting by randomly deactivating neurons during training, forcing the network to learn robust representations and reducing reliance on specific features. This project aims to investigate the impact of dropout on neural network performance and compare it with other benchmark regularization methods.

## Project Structure
- `models/`: Includes the implementations of different neural network architectures with dropout and other regularization techniques.
- `results/`: Contains the results obtained from training and evaluating the models.
- `main.ipynb`: Jupyter Notebook showcasing the project's code and analysis.

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
