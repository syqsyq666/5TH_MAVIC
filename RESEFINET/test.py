import re

import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms, models
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import cv2

# 项目根目录，便于统一管理路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

device_ids = [0]  # 按你的 GPU 编号修改，单卡可用 [0]
device = f'cuda:{device_ids[0]}' if torch.cuda.is_available() else 'cpu'

class InfDataset(Dataset):
    def __init__(self, img_folder, transform=None):
        self.imgs_folder = img_folder
        self.transform = transform
        self.img_paths = []

        img_path = self.imgs_folder + '/'
        img_list = os.listdir(img_path)
        img_list.sort()

        self.img_nums = len(img_list)

        for i in range(self.img_nums):
            img_name = img_path + img_list[i]
            self.img_paths.append(img_name)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx])
        if self.transform:
            img = self.transform(img)
        name = os.path.basename(self.img_paths[idx])  # "Gotcha16664030.png"

        match = re.search(r'\d+', name)
        if match:
            image_id = match.group()
        else:
            raise ValueError(f"无法从文件名 {name} 提取 image_id")
        return (img, image_id)

    def __len__(self):
        return self.img_nums

sar_transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert the tensor to PIL image
    transforms.Resize(224),   # Resize the image to the expected input size (224x224)
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale SAR image to 3-channel
    transforms.ToTensor(),    # Convert the image to a tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize based on SAR image's distribution
])

inf_transform = transforms.Compose(
    [transforms.ToPILImage(), transforms.Resize(224), transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# dataloader of the dataset（测试集路径：项目下 datasets/test）
img_folder = os.path.join(PROJECT_ROOT, 'datasets', 'test')
inf_dataset = InfDataset(img_folder, transform=inf_transform)
inf_dataloader = data.DataLoader(inf_dataset, batch_size=64, shuffle=False)


def _build_resnet101_10class():
    """与 norm_resnet50_SAR.py 中 SAR 分支一致：ResNet101，fc 改为 10 类"""
    model = models.resnet101(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, 10))
    return model


def _build_efficientnet_b0_10class():
    """与 efficient_SAR.py 中 SAR 分支一致：EfficientNet-B0，classifier 改为 10 类"""
    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(num_ftrs, 10))
    return model


def _load_model(path, model_builder, device):
    """支持两种保存格式：完整模型 或 仅 state_dict；state_dict 只加载形状匹配的键，避免结构不同报错"""
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, nn.Module):
        return ckpt
    if isinstance(ckpt, dict):
        if 'model_state_dict' in ckpt:
            state = ckpt['model_state_dict']
        elif 'model' in ckpt and isinstance(ckpt['model'], nn.Module):
            return ckpt['model']
        else:
            state = ckpt
        model = model_builder()
        # 只加载与当前模型形状一致的参数，跳过不匹配的（如 ResNeSt vs ResNet 结构差异）
        model_state = model.state_dict()
        matched = {k: v for k, v in state.items() if k in model_state and v.shape == model_state[k].shape}
        model.load_state_dict(matched, strict=False)
        return model
    raise TypeError(f"无法识别的 checkpoint 类型: {type(ckpt)}")


def test():
    # 模型权重路径：放在项目根目录下（可为完整模型或 state_dict）
    model_path_resnet = os.path.join(PROJECT_ROOT, 'checkpoints/resnet101/SAR_cross_domain_resnet50_epoch_22.pth')
    model_path_efficient = os.path.join(PROJECT_ROOT, 'checkpoints/efficientnet_b0/SAR_cross_domain_efficientB0_final.pth')

    model_SAR_model1 = _load_model(model_path_resnet, _build_resnet101_10class, device)
    model_SAR_model2 = _load_model(model_path_efficient, _build_efficientnet_b0_10class, device)

    model_SAR_model1.to(device)
    model_SAR_model2.to(device)

    image_id_list = []
    class_id_list = []
    score_list = []
    model_SAR_model1.eval()
    model_SAR_model2.eval()

    # 对模型输出进行 z-score 标准化
    def normalize_output(output):
        mean = output.mean(dim=0, keepdim=True)
        std = output.std(dim=0, keepdim=True)
        return (output - mean) / (std + 1e-5)  # 防止除以零
    with torch.no_grad():
        for batch_idx, (img, name) in tqdm(enumerate(inf_dataloader)):
            img = img.to(device)
            output_unlabeled_SAR_Resnet = normalize_output(model_SAR_model1(img))
            output_unlabeled_SAR_Eff_orignial = normalize_output(model_SAR_model2(img))
            output_unlabeled_SAR = torch.add(0.82 *output_unlabeled_SAR_Resnet,0.18 * output_unlabeled_SAR_Eff_orignial )
            score, pseudo_labeled = torch.max(output_unlabeled_SAR, 1)
            for i in range(len(name)):
                image_id_list.append(int(name[i]))
                class_id_list.append(pseudo_labeled[i].cpu().numpy())
                score_list.append(score[i].cpu().numpy())
    if not (len(image_id_list) == len(class_id_list) == len(score_list)):
        raise ValueError(
            f"❌ not match！ image_id_list={len(image_id_list)}, class_id_list={len(class_id_list)}, score_list={len(score_list)}")
    print("📌 type：")
    print(f"type(image_id_list) = {type(image_id_list)}")
    print(f"type(class_id_list) = {type(class_id_list)}")
    print(f"type(score_list) = {type(score_list)}")
    df = pd.DataFrame({'image_id': image_id_list,
                       'class_id': class_id_list,
                       'score': score_list})

    results_path = os.path.join(PROJECT_ROOT, 'out/results.csv')
    df.to_csv(results_path, mode='w', index=False, header=True)
    print(f'结果已保存至: {results_path}')


test()