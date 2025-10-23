import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.train import parallel_load_images
from utils.image_utils import list_image_files

from PIL import Image
import numpy as np
from skimage.metrics import  structural_similarity, peak_signal_noise_ratio

def name_path(pair):
    print(pair)
    name, path = pair.split(',')
    return name, Path(path)


def compute_mse_ssim_psnr(source_dir, method_dirs):
    results = {'Method': [], 'SSIM': [], 'PSNR': []}
    target_size = (256, 256)
    for method, path_dataset in method_dirs:

        ssim_list, psnr_list = [], []
        # 读取目标数据集中的所有图像文件

        if method.startswith('ref'):
            method_files = list(path_dataset.glob('*.png'))
            method_file_dict = {file.stem: file for file in method_files}

        else:
            method_files = list(path_dataset.glob('*to*.png'))
            method_file_dict = {file.stem.split('to')[0]: file for file in method_files}

        i = 0

        patterns = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        for pattern in patterns:
            for source_image_path in source_dir.glob(pattern):

                source_index_str = source_image_path.stem
                print(i)

                if source_index_str not in method_file_dict:
                    print("skip")
                    continue  # 如果找不到对应的图像文件，跳过

                method_image_path = method_file_dict[source_index_str]

                source_image = Image.open(source_image_path).convert('RGB').resize(target_size, Image.LANCZOS)
                method_image = Image.open(method_image_path).convert('RGB').resize(target_size, Image.LANCZOS)

                print(source_image_path)
                print(method_image_path)

                source_np = np.array(source_image)
                method_np = np.array(method_image)


                source_tensor = torch.from_numpy(source_np).float() / 255.0
                method_tensor = torch.from_numpy(method_np).float() / 255.0

                ssim = structural_similarity(source_tensor.numpy(), method_tensor.numpy(), channel_axis=-1, data_range=1)
                psnr = peak_signal_noise_ratio(source_tensor.numpy(), method_tensor.numpy())


                ssim_list.append(ssim)
                psnr_list.append(psnr)

                i += 1

        # 将结果存储到当前方法的结果中
        results['Method'].append(method)
        results['SSIM'].append(sum(ssim_list) / len(ssim_list))
        results['PSNR'].append(sum(psnr_list) / len(psnr_list))

    df = pd.DataFrame(results)

    return df

def main(args):
    datasets = {}

    source = args.source_dataset.name
    datasets[source] = parallel_load_images(args.source_dataset, list_image_files(args.source_dataset))

    if args.methods_dataset:
        args.methods_dataset = [name_path(pair) for pair in args.methods_dataset]

    print(args.methods_dataset)
    print(args.source_dataset)
    for method, path_dataset in args.methods_dataset:
        datasets[method] = parallel_load_images(path_dataset, list_image_files(path_dataset))


    # Compute  SSIM, PSNR
    df_mse_ssim_psnr = compute_mse_ssim_psnr(args.source_dataset, args.methods_dataset)

    df_result = pd.concat([ df_mse_ssim_psnr.set_index('Method')], axis=1).round(2)
    print(df_result)

    os.makedirs(args.output.parent, exist_ok=True)
    df_result.to_csv(args.output, index=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute metrics')
    parser.add_argument('--source_dataset', type=Path, default="dataset/Celeba/source", help='Dataset with real faces')

    parser.add_argument('--methods_dataset',
                        default=[
                            'Adv-Inversion-IDF-IR152,output/Adv-Inversion-IDF-Celeba/IR152',
                            'Adv-Inversion-IDF-IRSE50,output/Adv-Inversion-IDF-Celeba/IRSE50',
                            'Adv-Inversion-IDF-FaceNet,output/Adv-Inversion-IDF-Celeba/FaceNet',
                            'Adv-Inversion-IDF-MobileFace,output/Adv-Inversion-IDF-Celeba/MobileFace',
                        ],
                        nargs='+', help='Datasets after applying the method')

    parser.add_argument('--output', type=Path, default='logs/ssim_psnr_metric.csv', help='Folder for saving logs')
    args = parser.parse_args()

    main(args)
