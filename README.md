# Activation-Based Prototypes for Explainability in Semantic Segmentation of SAR Imagery

This repository contains the implementation of my Bachelor Thesis at TU Berlin. 

The project implements a framework utilizing prototypes and retrieving highest the matching ones for a query based on the activations through different neural network layers.

## Project Structure

```
ActivationPrototypes-SARSeg/
└── ActivationPrototypes-SARSeg 
│   ├── __init__.py            
│   └── thesis_utils  
│  
├── data                            <- Will have to be created and populated by you
│   └── BigEarthNet
│       ├── Encoded-BigEarthNet     <- LMDB created via rico-hdl
│       ├── prototypes              <- generated in `generating_prototypes` 
│       ├── Reference_Maps
│       └── metadata.parquet
│
├── models             
│
├── notebooks
│   ├── training_SkipConn           <- Training the U-Net
│   ├── generating_prototypes
│   ├── find_top_n                  <- Precalculate Prototype Retrieval
│   ├── top_matches_skipp_conn.csv  <- Precalculated as CSV
│   ├── qualitative_analysis
│   ├── quantitative_analysis
│   ├── noisy_results.csv           <- Results for `Continuity` Experiment
│   └── runtime 
│      
├── LICENSE
│
├── README.md 
│                        
├── pyproject.toml    
│                      
└── requirements.txt   
           
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
