# Adv-Inversion

Repository for the official implementation of our paper "Adv-Inversion: Stealthy Adversarial Attacks via GAN-Inversion for Facial Privacy Protection".

---

## Abstract

With the rapid advancement of deep face recognition (FR) systems, concerns over the unauthorized use of facial data have become increasingly serious. Although adversarial attacks have been employed to obscure identity information and protect user privacy, existing methods often struggle with degraded visual quality, low success rates in black-box attacks, and dependence on identity-specific training. To overcome these limitations, we introduce Adv-Inversion, a novel and stealthy adversarial attack technique for facial privacy protection. Our approach leverages an encoder-based GAN inversion framework, incorporating a redesigned feature style encoder to prioritize adversarial attacks over traditional editing tasks. By embedding adversarial perturbations iteratively into the feature tensor space, the method ensures high imperceptibility, robust attack transferability, and flexibility without the need for identity-specific training. Additionally, we introduce an **Identity Prior Feature Fusion Module** for identity-specific scenarios, enabling alignment between reconstructed and target faces while enhancing black-box attack success through an ensemble training strategy. Extensive experiments across two datasets, four open-source FR models on both face verification and face identification tasks, and two commercial FR APIs demonstrate that Adv-Inversion significantly outperforms related methods in both identity-free and identity-specific training scenarios, achieving state-of-the-art results in attack success rate and visual quality metrics. The approach also exhibits robustness against common adversarial defense methods. Multiple ablation studies further confirm the effectiveness of our model design.

---

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