import os
import sys
import typing as tp
from torchvision import transforms
from config import  get_fr_model, asr_calculation_txt
from dataset import adv_base_dataset
from torchvision.utils import save_image
from torch.utils.data import Subset
import argparse
from functools import wraps
from pathlib import Path
import numpy as np
import torchvision.transforms.functional as F
from utils.seed import seed_setter
from utils.time import bench_session
from models.Encoders import PostProcessModel_First
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch import nn
from models.Net import Net
from utils.bicubic import BicubicDownSample
TReturn = tp.TypeVar('TReturn', torch.Tensor, tuple[torch.Tensor, ...])

class AttackFSE:

    def __init__(self, args):
        self.args = args
        self.net = Net(self.args)  # 初始化网络模型
        self.normalize = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.downsample_256 = BicubicDownSample(factor=4)

        # self.post_process = PostProcessModel().to(self.args.device).eval()
        self.post_process = PostProcessModel_First().to(self.args.device).eval()  #Multi-GPU training loading
        state_dict = torch.load(self.args.pp_checkpoint)['model_state_dict']

        # If using DDP, remove the "module." prefix from the keys
        if 'module.' in next(iter(state_dict)):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        # Load the modified state_dict into the model
        self.post_process.load_state_dict(state_dict)
        ##########################
        # self.post_process.load_state_dict(torch.load(self.args.pp_checkpoint)['model_state_dict'])

    def random_transform(self,x):
        if torch.rand(1) > 0.5:
            x = torch.flip(x, [3])

        if torch.rand(1) > 0.5:
            scale_factor = torch.rand(1).item() * 0.2 + 0.9
            height, width = x.shape[2], x.shape[3]
            new_size = (int(height * scale_factor), int(width * scale_factor))
            x = F.interpolate(x, size=new_size, mode='bilinear', align_corners=True)

            if scale_factor < 1.0:
                padding_top = (height - new_size[0]) // 2
                padding_bottom = height - new_size[0] - padding_top
                padding_left = (width - new_size[1]) // 2
                padding_right = width - new_size[1] - padding_left
                x = F.pad(x, (padding_left, padding_right, padding_top, padding_bottom), mode='constant', value=0)
            elif scale_factor > 1.0:
                crop_top = (new_size[0] - height) // 2
                crop_bottom = crop_top + height
                crop_left = (new_size[1] - width) // 2
                crop_right = crop_left + width
                x = x[:, :, crop_top:crop_bottom, crop_left:crop_right]
        return x


    def face_targeted_attack_loss(self, face_embedding, orig_embedding, target_embedding):

        orig_cosine_sim = F.cosine_similarity(face_embedding, orig_embedding, dim=-1)
        target_cosine_sim = F.cosine_similarity(face_embedding, target_embedding, dim=-1)
        cosine_diff = target_cosine_sim-args.b*orig_cosine_sim

        loss = cosine_diff.sum()

        return loss

    # Adversarial Attack
    def adversarial_attack(self, latent: torch.Tensor, latent2: torch.Tensor, x_target: torch.Tensor, x_orig,classifier: dict,
                           classifier_scale: float, steps: int,di_true) -> torch.Tensor:
        B = x_target.shape[0]
        adv_latent = latent.clone().detach().requires_grad_(True)

        for step in range(steps):
            print(f'Step: {step + 1}/{steps}')

            x_in, _ = self.net.generator([latent2], input_is_latent=True, return_latents=False,
                                         start_layer=5, end_layer=8, layer_in=adv_latent)
            x_in = x_in.clip(-1, 1)

            if di_true==True:
                # print('di')
                x_in = self.random_transform(x_in)

            total_loss = 0
            for k, v in classifier.items():
                resize = torch.nn.AdaptiveAvgPool2d((112, 112)) if k != 'FaceNet' else torch.nn.AdaptiveAvgPool2d(
                    (160, 160))

                target_embeddings = v(resize(x_target)).reshape(B, -1)
                source_embeddings = v(resize(x_in)).reshape(B, -1)
                ori_embeddings = v(resize(x_orig)).reshape(B, -1)

                total_loss += self.face_targeted_attack_loss(source_embeddings, ori_embeddings,target_embeddings)

            total_loss = total_loss / len(classifier)
            total_loss.backward()
            grad = adv_latent.grad
            adv_latent = adv_latent + classifier_scale * grad
            adv_latent = adv_latent.detach().requires_grad_(True)

        return adv_latent


    def attack(self, image_source,image_target, **kwargs) -> torch.Tensor:
        with torch.no_grad():
            x_target = kwargs['x_target']
            classifier = kwargs['classifier']
            classifier_scale = kwargs['classifier_scale']
            di_true = kwargs['di_true']
            step = kwargs['step']

            device = self.args.device
            image_target = image_target.to(device)
            image_source = image_source.to(device)
            image_normalize_source=(image_source+1)/2
            image_normalize_target=(image_target+1)/2

            im_256_source = self.downsample_256(image_normalize_source)
            im_256_norm_source = self.normalize(im_256_source)

            im_256_target = self.downsample_256(image_normalize_target)
            im_256_norm_target = self.normalize(im_256_target)

            S_final, F_final = self.post_process(im_256_norm_source, im_256_norm_target)
        F_final = self.adversarial_attack(F_final, S_final, x_target,image_source, classifier, classifier_scale, steps=step,di_true=di_true)

        with torch.no_grad():
            adversarial_image, _ = self.net.generator([S_final], input_is_latent=True, return_latents=False,
                                            start_layer=5, end_layer=8, layer_in=F_final)

        final_image = adversarial_image.clip(-1, 1)
        return final_image

    @seed_setter
    @bench_session
    def inverseattack(self, image_source,image_target ,**kwargs) -> TReturn:

        final_image = self.attack(image_source, image_target,**kwargs)
        return final_image

    @wraps(inverseattack)
    def __call__(self, *args, **kwargs):
        return self.inverseattack(*args, **kwargs)


def get_parser():
    parser = argparse.ArgumentParser(description='stylegan2 setting')
    # StyleGAN2 setting
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--ckpt', type=str, default="pretrained_models/StyleGAN/ffhq.pt")
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--latent', type=int, default=512)
    parser.add_argument('--n_mlp', type=int, default=8)
    # Arguments
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--pp_checkpoint', type=str, default='pretrained_models/enhacend_fse.pth',help="Enhanced FSE Weight File Loading")
    return parser

def main(model_args, args):
    your_seed_value = 3447  # You can choose any integer as the seed.
    torch.manual_seed(your_seed_value)
    np.random.seed(your_seed_value)

    device = 'cuda:0'
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"文件夹 '{args.output_dir}' 已创建。")
    else:
        print(f"文件夹 '{args.output_dir}' 已存在。")

    transform = transforms.Compose([transforms.Resize((1024, 1024)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    dataset = adv_base_dataset(dir=args.input_dir, src=args.source,target=args.target,transform=transform, print_pairs=True)
    dataset = Subset(dataset, [x for x in range(len(dataset))])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=20, shuffle=False, num_workers=0)

    # attack_model_names = ['IR152']
    attack_model_names = ['IR152', 'IRSE50', 'FaceNet', 'MobileFace']
    attack_model_dict = {'IR152': get_fr_model('IR152'), 'IRSE50': get_fr_model('IRSE50'),
                         'FaceNet': get_fr_model('FaceNet'), 'MobileFace': get_fr_model('MobileFace')}

    # Assuming args.model is a single model name, not a list
    cos_sim_scores_dict = {model_name: [] for model_name in attack_model_names}

    attack_fse = AttackFSE(model_args)

    scores = []

    for attack_model_name in attack_model_names:

        attack_output_dir = os.path.join(args.output_dir, attack_model_name)
        if not os.path.exists(attack_output_dir):
            os.makedirs(attack_output_dir)
            print(f"文件夹 '{attack_output_dir}' 已创建。")
        else:
            print(f"文件夹 '{attack_output_dir}' 已存在。")

    for attack_model_name in attack_model_names:
        classifier = {k: v for k, v in attack_model_dict.items() if k != attack_model_name}
        attack_output_dir = os.path.join(args.output_dir, attack_model_name)
        attack_model = attack_model_dict[attack_model_name]
        resize = nn.AdaptiveAvgPool2d((112, 112)) if attack_model_name != 'FaceNet' else nn.AdaptiveAvgPool2d(
            (160, 160))

        for i, (image, tgt_image, img_name_pre, tgt_img_name_pre) in enumerate(dataloader):
            print('Number:',i)
            image = image.to(device)
            B=image.shape[0]
            tgt_image = tgt_image.to(device)
            ###
            final_image = attack_fse.inverseattack(image,tgt_image,
                                                   classifier=classifier,
                                                   classifier_scale=args.classifier_scale,
                                                   x_source=image,
                                                   x_target=tgt_image,
                                                   step=args.step,
                                                   di_true=args.di)

            for j in range(0,B):
                output_path = os.path.join(attack_output_dir, f'{img_name_pre[j]}to{tgt_img_name_pre[j]}.png')
                save_image(((final_image[j] + 1) / 2), output_path)

            feature1 = attack_model(resize(final_image)).reshape(image.shape[0], -1)
            feature2 = attack_model(resize(tgt_image)).reshape(image.shape[0], -1)
            feature3 = attack_model(resize(image)).reshape(image.shape[0], -1)
            score = F.cosine_similarity(feature1, feature2)
            score2 = F.cosine_similarity(feature1, feature3)
            print(f'score: {score}')
            print(f'score2: {score2}')

            cos_sim_scores_dict[attack_model_name] += score.tolist()

            formatted_score = [round(s.item(), 4) for s in score]
            scores.append(formatted_score)
            print("Image manipulation complete and saved.")

    print("score：", scores)
    output_txt = os.path.join(args.output_dir, "output.txt")
    asr_calculation_txt(cos_sim_scores_dict,output_txt)

if __name__ == "__main__":
    model_parser = get_parser()
    parser = argparse.ArgumentParser(description='attack evaluate')
    parser.add_argument('--input_dir', type=Path, default='dataset/Celeba', help='The directory of the clean images to be attacked')

    parser.add_argument('--source', type=Path, default='source', help='The directory of the Source face images to be attacked')
    parser.add_argument('--target', type=str, default='target', help='The directory of the Target face images')
    parser.add_argument('--output_dir', type=Path, default='output/Adv-Inversion-IDF-Celeba', help='The directory for final results')

    parser.add_argument('--b', type=float, default=0.4, help='Corresponding beta')
    parser.add_argument('--di', type=int,default=1,help='Whether to use the DIM method')
    parser.add_argument('--step', type=int, default=10, help='Number of iterations')
    parser.add_argument('--classifier_scale', type=int, default=60,help='Corresponding alpha')
    # Arguments for a set of experiments

    args, unknown1 = parser.parse_known_args()#
    model_args, unknown2 = model_parser.parse_known_args()

    unknown_args = set(unknown1) & set(unknown2)
    if unknown_args:
        file_ = sys.stderr
        print(f"Unknown arguments: {unknown_args}", file=file_)

        print("\nExpected arguments for the model:", file=file_) #
        model_parser.print_help(file=file_)

        print("\nExpected arguments for evaluate:", file=file_)
        parser.print_help(file=file_)

        sys.exit(1)

    main(model_args, args)