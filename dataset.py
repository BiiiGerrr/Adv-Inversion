from torch.utils.data import Dataset
import os
from PIL import Image


class adv_base_dataset(Dataset):
    def __init__(self, dir, src='',target='', transform=None,print_pairs=False) -> None:
        super().__init__()

        self.dir = dir
        self.src =src
        self.target =target
        self.img_names = sorted(os.listdir(os.path.join(dir, self.src)))
        self.tgt_img_names = sorted(os.listdir(os.path.join(dir, self.target)))
        # print(self.img_names)

        self.transform = transform
        self.print_pairs = print_pairs

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        # 获取原始图像路径和名称前缀
        img_name = self.img_names[index]
        img_path = os.path.join(self.dir, self.src, img_name)
        img_name_pre = os.path.splitext(img_name)[0]

        img = Image.open(img_path).convert('RGB')

        tgt_index = index % len(self.tgt_img_names)
        tgt_img_name = self.tgt_img_names[tgt_index]
        tgt_img_path = os.path.join(self.dir, self.target, tgt_img_name)
        tgt_img_name_pre = os.path.splitext(tgt_img_name)[0]

        tgt_img = Image.open(tgt_img_path).convert('RGB')


        if self.transform:
            img = self.transform(img)
            tgt_img = self.transform(tgt_img)

        if self.print_pairs:
            print(f'Loaded pair: {img_path}, {tgt_img_path}')

        return img, tgt_img, img_name_pre, tgt_img_name_pre

