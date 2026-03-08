# Activation-Based Prototypes for Explainability in Semantic Segmentation of SAR Imagery

This repository contains the implementation for the paper "Activation-Based Examples for Explainability in Semantic Segmentation of SAR Imagery".

The project implements a framework utilizing examples and retrieving the highest matching ones for a query based on the activations through different neural network layers.

## Installation

### 1) Create environment

```powershell
conda create -n sar_seg python=3.10 -y
conda activate sar_seg
```

### 2) Install package dependencies from `pyproject.toml`

CPU setup (default):

```powershell
pip install -e .
```

CUDA 11.8 setup (recommended for GPU training/inference):

```powershell
pip install --extra-index-url https://download.pytorch.org/whl/cu118 torch==2.7.1+cu118 torchvision==0.22.1+cu118
pip install --extra-index-url https://download.pytorch.org/whl/cu118 -e .
```

## Project Structure

```
ActivationExamples_SARSeg/
└── ActivationExamples_SARSeg 
│   ├── __init__.py            
│   └── examples_utils  
│  
├── data                            <- Will have to be created and populated by you
│   └── BigEarthNet
│       ├── Encoded-BigEarthNet     <- LMDB created via rico-hdl
│       ├── examples_subsample              <- generated in `generating_examples` 
│       ├── Reference_Maps
│       └── metadata.parquet
│
├── models             
│
├── notebooks
│   ├── training_SkipConn           <- Training the U-Net
│   ├── generating_examples
│   ├── find_top_n                  <- Precalculate Example Retrieval
│   ├── top_matches_skipp_conn.csv  <- Precalculated as CSV
│   ├── qualitative_analysis
│   ├── quantitative_analysis
│   ├── plots                       <- Stores the plots from experiments
│   ├── noisy_results.csv           <- Results for `Continuity` Experiment
│   └── runtime 
│      
├── LICENSE
│
├── README.md 
│                        
└── pyproject.toml                        
```

## References

1. **reBEN**: This dataset is a large-scale remote sensing dataset used in this project for training and evaluation.  
   - **Paper**: K. Clasen, L. Hackel, T. Burgert, G. Sumbul, B. Demir, V. Markl, "reBEN: Refined BigEarthNet Dataset for Remote Sensing Image Analysis", IEEE International Geoscience and Remote Sensing Symposium (IGARSS), 2025.  

2. **ConfigILM**: A general-purpose configurable library for combining image and language models for visual question answering.  
   - **Paper**: L. Hackel, K. Clasen, B. Demir, "ConfigILM: A General Purpose Configurable Library for Combining Image and Language Models for Visual Question Answering", SoftwareX 26 (2024): 101731.  

3. **Segmentation Models PyTorch**: This project utilizes functions from the [Segmentation Models PyTorch](https://github.com/qubvel-org/segmentation_models.pytorch) library.

## Datasets

### Getting and Loading Datasets as LMDB

To use the **reBEN dataset** for training or evaluation, follow these steps to download and prepare the dataset in LMDB format:

#### Step 1: Download the Datasets
- Visit the [BigEarthNet website](https://bigearth.net) to download the following:
  - S1/S2 satellite data
  - Reference maps
  - metadata parquet

#### Step 2: Convert to LMDB format using `rico-hdl`
- Install `rico-hdl` by following the instructions at [rico-hdl GitHub repository](https://github.com/kai-tub/rico-hdl).

- Store the required files according to the [Project Structure](#project-structure) section. described above
