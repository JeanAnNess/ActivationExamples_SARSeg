# Activation-Based Examples for Explainability in Semantic Segmentation of SAR Imagery

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
│
├── ActivationExamples_SARSeg/
│   ├── __init__.py                     
│   ├── config.py                       # Constants, color maps, class mappings, layer shapes
│   ├── examples_utils.py               # re-exports
│   │
│   ├── activations/
│   │   ├── hooks.py                    # Hook registration, activation extraction (U-Net & DeepLabV3+)
│   │   ├── storage.py                  # Store activations in LMDB database
│   │   └── retrieval.py                # Cosine/euclidean similarity search, top-N retrieval
│   │
│   ├── data/
│   │   ├── lmdb.py                     # LMDB key matching, image/mask loading
│   │   └── dataset.py                  # PyTorch Dataset
│   │
│   ├── models/
│   │   ├── unet.py                     # CustomUnetSkipConn, factories, pretrained loading
│   │   ├── deeplabv3plus.py            # DeepLabV3+ factories, pretrained loading
│   │   └── registry.py                 # layer shapes/scale factors per architecture
│   │
│   ├── training/
│   │   └── trainer.py                  # Training loop, evaluation, inference, checkpointing
│   │
│   ├── analysis/
│   │   └── overlap.py                  # Pairwise/aggregate overlap analysis, IoU-based matching
│   │
│   └── visualization/
│       └── plotting.py                 # Grid visualizations, overlap heatmaps
│
├── data/BigEarthNet/                   # Will have to be created and populated by you
│   ├── Encoded-BigEarthNet/            # LMDB created via rico-hdl
│   ├── examples_subsample/             # Generated activation examples (U-Net)
│   ├── examples_subsample_deeplabv3p/  # Generated activation examples (DeepLabV3+)
│   ├── Reference_Maps/
│   └── metadata.parquet
│
├── models/
│
├── notebooks/
│   ├── training_SkipConn.ipynb               # U-Net training
│   ├── training_DeepLabV3Plus.ipynb          # DeepLabV3+ training
│   ├── generating_examples.ipynb             # Activation extraction (U-Net)
│   ├── generating_examples_deeplab.ipynb     # Activation extraction (DeepLabV3+)
│   ├── find_top_n.ipynb                      # Precompute top-N similarity (U-Net)
│   ├── find_top_n_deeplab.ipynb              # Precompute top-N similarity (DeepLabV3+)
│   ├── quantitative_analysis.ipynb           # Correctness/faithfulness metrics
│   ├── qualitative_analysis.ipynb            # Visualizations (U-Net)
│   ├── qualitative_analysis_deeplab.ipynb    # Visualizations (DeepLabV3+)
│   └── runtime.ipynb                         # Runtime profiling
│
├── README.md
├── pyproject.toml
└── LICENSE
```

## Pipeline
Training, activation extraction and precomputing similarities exists for two architectures: U-Net (default) and DeepLabV3+ (indicated with suffix `_deeplab`).

1. **Training**: `notebooks/training_SkipConn.ipynb` or `training_DeepLabV3Plus.ipynb`
2. **Activation Extraction**: `generating_examples.ipynb` stores intermediate activations into LMDB
3. **Precompute Similarities**: `find_top_n.ipynb` computes top-N matching regions via sliding window cosine similarity
4. **Quantitative Analysis**: `quantitative_analysis.ipynb`: correctness, continuity, overlap metrics for U-Net
5. **Qualitative Analysis**: `qualitative_analysis.ipynb`: visual inspection of retrieved examples

### Outputs
- `notebooks/plots/`: generated figures (overlap matrices, qualitative grids)
- `notebooks/top_matches_skipp_conn.csv`: precomputed matches for U-Net
- `notebooks/top_matches_deeplabv3p.csv`: precomputed matches for DeepLabV3+
- `data/BigEarthNet/examples_subsample*/`: LMDB databases of stored activations

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
- Store the required files according to the [Project Structure](#project-structure) section.
