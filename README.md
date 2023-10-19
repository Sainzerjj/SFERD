# Reducing Spatial Fitting Error in Distillation of Denoising Diffusion Models (SFERD)

## Official implementation for paper: "Reducing Spatial Fitting Error in Distillation of Denoising Diffusion Models"
## Abstract
We propose **S**patial **F**itting-**E**rror **R**eduction **D**istillation model ($\textbf{SFERD}$). SFERD utilizes attention guidance from the teacher model and a designed semantic gradient predictor to reduce the student's fitting error. Empirically, our proposed model facilitates high-quality sample generation in a few function evaluations (2~4 steps). To gain insight from our exploration of the self-attention maps of diffusion models and for detailed explanations, please see our Paper and Project Page.

## Pipeline
![process](https://github.com/Sainzerjj/SFERD/blob/main/imgs/distillation.png)  
![process](https://github.com/Sainzerjj/SFERD/blob/main/imgs/DAE.png)
![process](https://github.com/Sainzerjj/SFERD/blob/main/imgs/DSE.png)  

## Requirements
A suitable conda environment named SFERD can be created and activated with:
```
conda env create -f environment.yaml
conda activate SFERD
```
## Training

## Sampling

## Acknowledgements
This implementation is based on the repo from [openai/guided-diffusion](https://github.com/openai/guided-diffusion) and [openai/consistency_models](https://github.com/openai/consistency_models).

