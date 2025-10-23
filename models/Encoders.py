import argparse

import clip
import torch
import torch.nn as nn
from torch.nn import Linear, LayerNorm, LeakyReLU, Sequential
from torchvision import transforms as T
from models.attension import SpatialTransformer
from models.Net import FeatureEncoderMult, IBasicBlock, conv1x1,FeatureEncoderMult2
from models.stylegan2.model import PixelNorm


class ModulationModule(nn.Module):
    def __init__(self, layernum, last=False, inp=512, middle=512):
        super().__init__()
        self.layernum = layernum
        self.last = last
        self.fc = Linear(512, 512)
        self.norm = LayerNorm([self.layernum, 512], elementwise_affine=False)
        self.gamma_function = Sequential(Linear(inp, middle), LayerNorm([middle]), LeakyReLU(), Linear(middle, 512))
        self.beta_function = Sequential(Linear(inp, middle), LayerNorm([middle]), LeakyReLU(), Linear(middle, 512))
        self.leakyrelu = LeakyReLU()

    def forward(self, x, embedding):
        x = self.fc(x)
        x = self.norm(x)
        gamma = self.gamma_function(embedding)
        beta = self.beta_function(embedding)
        out = x * (1 + gamma) + beta
        if not self.last:
            out = self.leakyrelu(out)
        return out


class FeatureiResnet(nn.Module):
    def __init__(self, blocks, inplanes=1024):
        super().__init__()

        self.res_blocks = {}

        for n, block in enumerate(blocks, start=1):
            planes, num_blocks = block

            for k in range(1, num_blocks + 1):
                downsample = None
                if inplanes != planes:
                    downsample = nn.Sequential(conv1x1(inplanes, planes, 1), nn.BatchNorm2d(planes, eps=1e-05, ), )

                self.res_blocks[f'res_block_{n}_{k}'] = IBasicBlock(inplanes, planes, 1, downsample, 1, 64, 1)
                inplanes = planes

        self.res_blocks = nn.ModuleDict(self.res_blocks)

    def forward(self, x):
        for module in self.res_blocks.values():
            x = module(x)
        return x


class PostProcessModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_face = FeatureEncoderMult(fs_layers=[9], opts=argparse.Namespace(
            **{'arcface_model_path': "pretrained_models/ArcFace/backbone_ir50.pth"}))

        self.latent_avg = torch.load('pretrained_models/PostProcess/latent_avg.pt', map_location=torch.device('cuda'))
        self.to_feature = FeatureiResnet([[1024, 2], [768, 2], [512, 2]])

        self.to_latent_1 = nn.ModuleList([ModulationModule(18, i == 4) for i in range(5)])
        self.to_latent_2 = nn.ModuleList([ModulationModule(18, i == 4) for i in range(5)])
        self.pixelnorm = PixelNorm()

    def forward(self, source, target):
        s_face, [f_face] = self.encoder_face(source)
        s_hair, [f_hair] = self.encoder_face(target)

        dt_latent_face = self.pixelnorm(s_face)
        dt_latent_hair = self.pixelnorm(s_hair)

        for mod_module in self.to_latent_1:
            dt_latent_face = mod_module(dt_latent_face, s_hair)

        for mod_module in self.to_latent_2:
            dt_latent_hair = mod_module(dt_latent_hair, s_face)

        finall_s = self.latent_avg + 0.1 * (dt_latent_face + dt_latent_hair)

        cat_f = torch.cat((f_face, f_hair), dim=1)
        finall_f = self.to_feature(cat_f)

        return finall_s, finall_f

class PostProcessModel_attension(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_face = FeatureEncoderMult(fs_layers=[9], opts=argparse.Namespace(
            **{'arcface_model_path': "pretrained_models/ArcFace/backbone_ir50.pth"}))

        self.latent_avg = torch.load('pretrained_models/PostProcess/latent_avg.pt', map_location=torch.device('cuda'))
        self.to_feature = SpatialTransformer(in_channels=512)  # 初始化特征提取器

        self.to_latent_1 = nn.ModuleList([ModulationModule(18, i == 4) for i in range(5)])
        self.to_latent_2 = nn.ModuleList([ModulationModule(18, i == 4) for i in range(5)])
        self.pixelnorm = PixelNorm()

    def forward(self, source, target):
        s_source, [f_source] = self.encoder_face(source)  # 编码源图像
        s_target, [f_target] = self.encoder_face(target)  # 编码目标图像


        dt_latent_source = self.pixelnorm(s_source)  # 归一化人脸潜在表示
        dt_latent_target = self.pixelnorm(s_target)  # 归一化头发潜在表示

        for mod_module in self.to_latent_1:  # 遍历调制模块
            dt_latent_source = mod_module(dt_latent_source, s_target)  # 应用调制

        for mod_module in self.to_latent_2:  # 遍历调制模块
            dt_latent_target = mod_module(dt_latent_target, s_source)  # 应用调制
        finall_s = self.latent_avg + 0.1 * (dt_latent_source + dt_latent_target)  # 计算最终潜在表示
        # finall_s = self.latent_avg + 0.2 * (dt_latent_source + dt_latent_target)  # 计算最终潜在表示

        # cat_f = torch.cat((f_face, f_hair), dim=1)  # 连接人脸和头发特征
        b = f_target.size(0)  # 获取 batch 大小
        f_target_2 = f_target.view(b, 512, 64 * 64).permute(0, 2, 1)

        finall_f = self.to_feature(f_source, f_target_2)  # 提取最终特征

        # finall_f=0.95*finall_f+0.05*f_target
        return finall_s, finall_f  # 返回最终潜在表示和特征

class PostProcessModel_First(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_face = FeatureEncoderMult(fs_layers=[9], opts=argparse.Namespace(
            **{'arcface_model_path': "pretrained_models/ArcFace/backbone_ir50.pth"}))

        self.latent_avg = torch.load('pretrained_models/latent_avg.pt', map_location=torch.device('cuda'))
        self.to_feature = FeatureiResnet([[1024, 2], [768, 2], [512, 2]])

        self.to_latent = nn.Sequential(nn.Linear(1024, 1024), nn.LayerNorm([1024]), nn.LeakyReLU(),
                                       nn.Linear(1024, 512))  # 初始化像素归一化
        self.pixelnorm = PixelNorm()

    def forward(self, source, target):
        s_face, [f_face] = self.encoder_face(source)
        return self.latent_avg + s_face, f_face   # 返回潜在平均值和特征
        # return finall_s, finall_f

class ClipModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip_model, _ = clip.load("ViT-B/32", device="cuda")
        self.transform = T.Compose(
            [T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))]
        )
        self.face_pool = torch.nn.AdaptiveAvgPool2d((224, 224))

        for param in self.clip_model.parameters():
            param.requires_grad = False

    def forward(self, image_tensor):
        if not image_tensor.is_cuda:
            image_tensor = image_tensor.to("cuda")
        if image_tensor.dtype == torch.uint8:
            image_tensor = image_tensor / 255

        resized_tensor = self.face_pool(image_tensor)
        renormed_tensor = self.transform(resized_tensor)
        return self.clip_model.encode_image(renormed_tensor)
