# Reducing Spatial Fitting Error in Distillation of Denoising Diffusion Models (SFERD)

## Official implementation for paper: "Reducing Spatial Fitting Error in Distillation of Denoising Diffusion Models" (AAAI 2024)
[arxiv](https://arxiv.org/pdf/2311.03830.pdf)
## Abstract
We propose **S**patial **F**itting-**E**rror **R**eduction **D**istillation model ($\textbf{SFERD}$). SFERD utilizes attention guidance from the teacher model and a designed semantic gradient predictor to reduce the student's fitting error. Empirically, our proposed model facilitates high-quality sample generation in a few function evaluations (1~4 steps). **Our method can be applied to existing mainstream Diffusion Distillation models ([Consistency Distillation](https://arxiv.org/pdf/2303.01469.pdf), [Progress Distallation](https://arxiv.org/pdf/2202.00512.pdf) for better performance**.
To gain insight from our exploration of the self-attention maps of diffusion models and for detailed explanations, please see our Paper and Project Page.

## Pipeline
![distillation](https://github.com/Sainzerjj/SFERD/blob/main/imgs/distillation.png) 
![DAG](https://github.com/Sainzerjj/SFERD/blob/main/imgs/DAG.png)
![DSE](https://github.com/Sainzerjj/SFERD/blob/main/imgs/DSE.png)  

## Requirements
A suitable conda environment named SFERD can be created and activated with:
```
conda env create -f environment.yml
conda activate SFERD
```
## Illustration
We provide the main core code implementation of the SFERD model, which includes network design for the teacher model with attention guidance (`./unet/teacher_unet.py`), the student model with semantic gradient predictor (`./unet/student_unet.py`), the implementation of the diffusion distillation training process (`./diffusion/gaussian_diffusion.py`), trainer defination file (`./diffusion/train_utils.py`), the main file for distillation training (`train_diffusion_distillation.py`) and sampling(`sample.py`). 

Specially, the main work of `./unet/teacher_unet.py` is extracting the attention map of the middle or decoder blocks in diffusion model. The main work of `./unet/student_unet.py` is adding semantic encoder module, gradient predictor module and latent diffusion module, and futher incorporating them into training with the trained distillation student model. The main work of `./diffusion/gaussian_diffusion.py` is achieving attention guidance method based on teacher model, reformulating training loss objective with semantic gradient predictor, training diffusion distillation model, training latent diffusion and applying necessary diffusion process(including inference, forward, noise schedule setting.)

The detailed code will come soon !!!

## Acknowledgements
This implementation is based on the repo from [openai/guided-diffusion](https://github.com/openai/guided-diffusion) and [openai/consistency_models](https://github.com/openai/consistency_models).

