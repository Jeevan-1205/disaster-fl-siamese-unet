# Federated Disaster Damage Segmentation using Siamese U-Net

## Overview

This project explores **Federated Learning (FL)** for disaster damage segmentation using a **Siamese U-Net** architecture.

The goal is to detect and segment damaged regions by comparing **pre-disaster** and **post-disaster** satellite images while simulating decentralized training across multiple disaster clients.

This project evaluates multiple training approaches and compares their performance under heterogeneous client distributions.

---

## Objectives

* Build a disaster damage segmentation pipeline
* Implement and compare Federated Learning strategies
* Study performance under decentralized client settings
* Compare centralized and federated optimization approaches

---

## Methods Implemented

### 1. Centralized Training

All client data is merged and trained together.

### 2. Local Only Training

Each client trains independently without aggregation.

### 3. Federated Averaging (FedAvg)

Clients train locally and periodically aggregate model parameters.

### 4. Federated Proximal (FedProx)

Extension of FedAvg with proximal regularization for heterogeneous clients.

---

## Model Architecture

Architecture:
**Siamese U-Net**

Input:

* Pre-disaster image
* Post-disaster image

Output:

* Binary segmentation mask

Loss Function:

* BCEWithLogitsLoss

Optimizer:

* Adam

Learning Rate:

* 1e-4

---

## Dataset

Dataset Type:
Synthetic realistic disaster segmentation dataset

Image Resolution:
128 × 128

Clients:

* Earthquake
* Flood
* Wildfire
* Hurricane

Data Structure:

data/

├── earthquake/

├── flood/

├── wildfire/

└── hurricane/

Each sample contains:

* pre image
* post image
* damage mask

---

## Experiments

Training Environment:

* Ubuntu (WSL)
* VS Code
* PyTorch

Evaluation Metrics:

* IoU
* Dice Score
* Precision
* Recall

---

## Results

| Method      | IoU   | Dice  | Precision | Recall |
| ----------- | ----- | ----- | --------- | ------ |
| Centralized | 0.989 | 0.994 | 0.998     | 0.990  |
| Local Only  | 0.977 | 0.988 | 0.987     | 0.990  |
| FedProx     | 0.906 | 0.951 | 0.909     | 0.996  |
| FedAvg      | 0.902 | 0.948 | 0.904     | 0.997  |

---

## Key Findings

* Centralized training achieved the strongest overall performance.
* Local-only training achieved near-centralized performance.
* Federated methods maintained competitive performance under decentralized constraints.
* FedProx slightly improved FedAvg under heterogeneous clients.

---

## Project Structure

notebooks/

src/

data/

README.md

RESULTS.md

---

## Run

Train Federated:

python -m notebooks.fed_round

Evaluate:

python -m notebooks.evaluate_fed

Train Centralized:

python -m notebooks.train_centralized_realistic

Evaluate Centralized:

python -m notebooks.evaluate_centralized_realistic

---

## Future Work

* Evaluate on xBD disaster dataset
* Experiment with FedPer
* Add communication cost analysis
* Improve client heterogeneity

---

## Author

Jeevan Prakash Meghwal

B.Tech Computer Science Engineering

Bennett University
