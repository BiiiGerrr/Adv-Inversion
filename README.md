# Adv-Inversion

Repository for the official implementation of our paper "Adv-Inversion: Stealthy Adversarial Attacks via GAN-Inversion for Facial Privacy Protection".

This repository contains the official implementation of Adv-Inversion, a novel framework for generating identity-preserving adversarial face samples via GAN inversion, designed to enhance facial privacy protection against unauthorized recognition systems.

## Installation

#### Build Environment

1.Create a new conda environment
   ```bash
   conda create -n advinversion python=3.10
   conda activate advinversion
   pip install -r requirements.txt
   ```
2.Download Checkpoints

We use **IR152**, **IRSE50**, **FaceNet**, and **MobileFace** model checkpoints provided by [AMT-GAN](https://github.com/CGCL-codes/AMT-GAN). The Google Drive link to these checkpoints is [here](https://drive.google.com/file/d/1Vuek5-YTZlYGoeoqyM5DlvnaXMeii4O8/view). After downloading, create the `pretrained_models` folder:
   ```bash
   cd Adv-Inversion-main
   mkdir pretrained_models    
   ```
Place the `.pth` files in the `pretrained_models` folder (you can unzip the files from `assets.zip` to get them).

Additional StyleGAN and ArcFace backbone weights are necessary. Please download them from the provided [Google Drive link](https://drive.google.com/file/d/1Ugf4yB9NeLtbKaxrJGPrrLZLH1YxpC_s/view?usp=sharing), unzip them, and place them in the `pretrained_models` folder.

Download the **enhanced FSE model weights** from the given [Google Drive link](https://drive.google.com/file/d/1OfeeV6AVGec6bkkKmw-nzETr9bJeLDPk/view?usp=sharing) and place them in the `pretrained_models` folder.

3.Download Datasets

For our experiments, we use FFHQ and CelebA-HQ datasets for evaluation. Due to ownership restrictions, we cannot provide direct access to these datasets. Please download them manually. We only provide the target face for your use. Ensure that the source face is placed in the `dataset/Celeba/source` or `dataset/FFHQ/source` directory.

## Usage

To perform the adversarial attack, use the following command (Place the source face in the dataset/Celeba/source directory, and adjust the batch size when the available GPU memory is insufficient.):
   ```bash
   python attack_IDF.py
   ```
To evaluate the attack results using FID metric:
   ```bash
   python eval_fid_metric.py
   ```
To evaluate the attack results using SSIM and PSNR metrics:
   ```bash
   python eval_ssim_psnr_metric.py
   ```

##Citation

To be added after paper acceptance.



If you have any questions, please contact [wanghb69@mail2.sysu.edu.cn]
