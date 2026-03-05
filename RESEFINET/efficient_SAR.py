import torch
import torch.nn as nn
import sys
import os
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from itertools import cycle, zip_longest

# 补充缺失的损失函数定义
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, weight=self.alpha, reduction='mean')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss

class da_loss(nn.Module):
    def __init__(self):
        super(da_loss, self).__init__()

    def forward(self, feat1, feat2):
        return F.mse_loss(feat1.mean(dim=0), feat2.mean(dim=0))

# 项目根目录配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints', 'efficientnet_b0')
from sklearn.model_selection import train_test_split
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import Dataset, Subset, DataLoader

def _get_devices():
    n = torch.cuda.device_count()
    if n >= 2:
        return [0, 1], 'cuda:0', 'cuda:1'
    if n == 1:
        return [0], 'cuda:0', 'cuda:0'
    return [0], 'cpu', 'cpu'

device_ids, device_1, device_2 = _get_devices()

# 数据集定义（保持不变）
class Datasets(Dataset):
    def __init__(self, datasetA, datasetB):
        self.datasetA = datasetA
        self.datasetB = datasetB

    def __getitem__(self, index):
        (xA, lA) = (self.datasetA[index][0], torch.tensor(self.datasetA[index][1]))
        (xB, lB) = (self.datasetB[index][0], torch.tensor(self.datasetB[index][1]))
        return (xA, lA), (xB, lB)

    def __len__(self):
        return len(self.datasetA)

class Unlabeled_Datasets(Dataset):
    def __init__(self, datasetA, datasetB, transform=None):
        self.datasetA = datasetA
        self.datasetB = datasetB
        self.transform = transform

    def __getitem__(self, index):
        xA = self.datasetA[index]
        xB = self.datasetB[index]
        if isinstance(xA, tuple):
            xA = xA[0]
        if isinstance(xB, tuple):
            xB = xB[0]

        if isinstance(xA, str):
            xA = Image.open(xA).convert("RGB")
        if isinstance(xB, str):
            xB = Image.open(xB).convert("RGB")

        if self.transform:
            xA = self.transform(xA)
            xB = self.transform(xB)

        if not isinstance(xA, torch.Tensor):
            xA = transforms.ToTensor()(xA)
        if not isinstance(xB, torch.Tensor):
            xB = transforms.ToTensor()(xB)

        return xA, xB
    
    def __len__(self):
        return len(self.datasetA)

class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = sorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image

def prepare_data_loaders(
        eo_path, sar_path, batch_size=60, test_size=0.2, num_workers=5, eo_transform=None, sar_transform=None):
    train_data_EO = torchvision.datasets.ImageFolder(root=eo_path, transform=eo_transform)
    train_data_SAR = torchvision.datasets.ImageFolder(root=sar_path, transform=sar_transform)

    targets = train_data_EO.targets
    indices = np.arange(len(targets))
    labeled_indices, unlabeled_indices = train_test_split(
        indices, test_size=test_size, stratify=targets
    )

    train_data_EO_labeled = Subset(train_data_EO, labeled_indices)
    train_data_EO_unlabeled = Subset(train_data_EO, unlabeled_indices)
    train_data_SAR_labeled = Subset(train_data_SAR, labeled_indices)
    train_data_SAR_unlabeled = Subset(train_data_SAR, unlabeled_indices)

    train_dataset_size_EO = len(train_data_EO)
    train_dataset_size_SAR = len(train_data_SAR)

    # 加权采样器
    y_train_EO = [train_data_EO_labeled.dataset.targets[i] for i in train_data_EO_labeled.indices]
    assert len(y_train_EO) == len(train_data_EO_labeled), "标签数量与数据数量不一致！"
    y_arr = np.array(y_train_EO)
    unique_classes = np.unique(y_arr)
    class_sample_count = np.array([len(np.where(y_arr == t)[0]) for t in unique_classes])
    weight = 1.0 / class_sample_count
    label_to_idx = {int(u): i for i, u in enumerate(unique_classes)}
    samples_weight = np.array([weight[label_to_idx[t]] for t in y_train_EO], dtype=np.float64)
    samples_weight = torch.from_numpy(samples_weight)

    sampler = WeightedRandomSampler(
        samples_weight.type(torch.DoubleTensor), len(samples_weight), replacement=True
    )

    train_dataset = Datasets(train_data_EO_labeled, train_data_SAR_labeled)
    unlabeled_dataset = Unlabeled_Datasets(train_data_EO_unlabeled, train_data_SAR_unlabeled)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, drop_last=True
    )
    unlabeled_train_loader = DataLoader(
        unlabeled_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True
    )

    return train_loader, unlabeled_train_loader, train_dataset_size_EO, train_dataset_size_SAR

def _load_local_pretrained(model, pth_path, device='cpu'):
    if not os.path.isfile(pth_path):
        # 增加错误提示，方便排查路径问题
        raise FileNotFoundError(f"预训练模型文件不存在: {pth_path}")
    ckpt = torch.load(pth_path, map_location=device)
    state = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else (ckpt.state_dict() if isinstance(ckpt, nn.Module) else ckpt)
    if not isinstance(state, dict):
        raise ValueError(f"预训练模型文件格式错误，无法提取state_dict: {pth_path}")
    model.load_state_dict(state, strict=False)
    print(f"  已从 {os.path.basename(pth_path)} 加载预训练 backbone（classifier 保持 10 类）")

def train(train_loader, unlabeled_train_loader, device_1, device_2, batch_size=60, total_epochs=30):
    # 模型初始化
    try:
        model_EO = torchvision.models.efficientnet_b0(weights=None)
    except TypeError:
        model_EO = torchvision.models.efficientnet_b0(pretrained=False)
    num_ftrs_EO = model_EO.classifier[1].in_features
    model_EO.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_ftrs_EO, 10))
    
    # ========== 核心修改1：修改预训练模型路径 ==========
    pretrained_path = '/root/autodl-tmp/CSRN_PBVS2025-main/efficientnet-b0-355c32eb.pth'
    _load_local_pretrained(model_EO, pretrained_path, device_1)

    try:
        model_SAR = torchvision.models.efficientnet_b0(weights=None)
    except TypeError:
        model_SAR = torchvision.models.efficientnet_b0(pretrained=False)
    num_ftrs_SAR = model_SAR.classifier[1].in_features
    model_SAR.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_ftrs_SAR, 10))
    
    # ========== 核心修改2：SAR模型也加载相同的预训练权重 ==========
    _load_local_pretrained(model_SAR, pretrained_path, device_2)

    for param in model_EO.parameters():
        param.requires_grad = True
    for param in model_SAR.parameters():
        param.requires_grad = True
    
    alpha_t = torch.tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]).to(device_1)
    criterion_ce = FocalLoss(alpha_t, 2)
    criterion_da = da_loss()
    alpha = 0.8
    beta = 0.2

    # 特征钩子
    activation_EO = {}
    def getActivation_EO(name):
        def hook(model, input, output):
            activation_EO[name] = output.detach()
        return hook

    activation_SAR = {}
    def getActivation_SAR(name):
        def hook(model, input, output):
            activation_SAR[name] = output.detach()
        return hook

    h1_EO = model_EO.features[8].register_forward_hook(getActivation_EO('8'))
    h1_SAR = model_SAR.features[8].register_forward_hook(getActivation_SAR('8'))

    model_EO.to(device_1)
    model_SAR.to(device_2)

    optim_EO = optim.Adam(model_EO.parameters(), lr=0.003)
    optim_SAR = optim.Adam(model_SAR.parameters(), lr=0.003)
    scheduler_EO = ReduceLROnPlateau(optim_EO, 'min', patience=7)
    scheduler_SAR = ReduceLROnPlateau(optim_SAR, 'min', patience=7)

    # ========== 核心修改1：配置每100个batch输出日志 ==========
    batch_log_interval = 100  # 每100个batch输出一次汇总日志
    global_batch_count = 0     # 全局batch计数器（跨epoch累计）
    batch_accum_loss_EO = 0.0  # 累计100个batch的EO损失
    batch_accum_loss_SAR = 0.0 # 累计100个batch的SAR损失
    batch_accum_correct_EO = 0 # 累计100个batch的EO正确数
    batch_accum_correct_SAR = 0# 累计100个batch的SAR正确数
    batch_accum_total_EO = 0   # 累计100个batch的EO样本数
    batch_accum_total_SAR = 0  # 累计100个batch的SAR样本数

    print("Starting training on EO and SAR data")
    print(f"\n🚀 Starting Training (Total Epochs: {total_epochs}, Batch Log Interval: {batch_log_interval} batches)...\n")
    
    for epoch in tqdm(range(total_epochs), desc="Epoch Progress", unit="epoch"):  
        epoch_train_loss_EO, epoch_train_loss_SAR = 0.0, 0.0
        epoch_correct_EO, epoch_correct_SAR, epoch_total_EO, epoch_total_SAR = 0.0, 0.0, 0.0, 0.0

        train_loader_iter = zip_longest(train_loader, unlabeled_train_loader, fillvalue=None)
        progress_bar = tqdm(train_loader_iter, desc=f"Epoch {epoch + 1}", unit="batch", leave=False)

        for (data_labeled, data_unlabeled) in progress_bar:
            if data_labeled is None or data_unlabeled is None:
                continue
            
            # ========== 核心修改2：更新全局batch计数器 ==========
            global_batch_count += 1

            # 数据加载（保持不变）
            (inputs_EO, labels_EO), (inputs_SAR, labels_SAR) = data_labeled
            inputs_EO, labels_EO = inputs_EO.to(device_1), labels_EO.to(device_1)
            inputs_SAR, labels_SAR = inputs_SAR.to(device_2), labels_SAR.to(device_2)
            inputs_EO_unlabeled, inputs_SAR_unlabeled = data_unlabeled
            inputs_EO_unlabeled = inputs_EO_unlabeled.to(device_1)
            inputs_SAR_unlabeled = inputs_SAR_unlabeled.to(device_2)

            # 前向传播（保持不变）
            outputs_EO = model_EO(inputs_EO)
            outputs_SAR = model_SAR(inputs_SAR).to(device_1)
            h1 = []
            h1.append(activation_EO['8'].to(device_1))
            h1.append(activation_SAR['8'].to(device_1))
            outputs_EO_unlabeled = model_EO(inputs_EO_unlabeled)
            outputs_SAR_unlabeled = model_SAR(inputs_SAR_unlabeled)
            h1.append(activation_EO['8'].to(device_1))
            h1.append(activation_SAR['8'].to(device_1))

            # 损失计算（保持不变）
            loss_ce_EO = criterion_ce(outputs_EO.to(device_1), labels_EO.to(device_1))
            loss_ce_SAR = criterion_ce(outputs_SAR.to(device_1), labels_SAR.to(device_1))
            loss_da = criterion_da(h1[0].to(device_1), h1[1].to(device_1)).to(device_1)
            loss_da_unlabeled = criterion_da(h1[2].to(device_1), h1[3].to(device_1)).to(device_1)
            loss_EO = loss_ce_EO + loss_ce_SAR + ((alpha * loss_da) + (beta * loss_da_unlabeled))
            loss_SAR = loss_ce_SAR + loss_ce_EO + ((alpha * loss_da) + (beta * loss_da_unlabeled))

            # 反向传播（保持不变）
            optim_EO.zero_grad()
            loss_EO.backward(retain_graph=True, inputs=list(model_EO.parameters()))
            optim_EO.step()
            optim_SAR.zero_grad()
            loss_SAR.backward(inputs=list(model_SAR.parameters()))
            optim_SAR.step()

            # 精度计算（保持不变）
            predictions_EO = outputs_EO.argmax(dim=1, keepdim=True).squeeze()
            correct_EO = (predictions_EO == labels_EO).sum().item()
            total_EO = labels_EO.size(0)

            predictions_SAR = outputs_SAR.argmax(dim=1, keepdim=True).squeeze()
            predictions_SAR = predictions_SAR.to(device_2)
            labels_SAR = labels_SAR.to(device_2)
            correct_SAR = (predictions_SAR == labels_SAR).sum().item()
            total_SAR = labels_SAR.size(0)

            # ========== 核心修改3：累加当前batch的指标 ==========
            batch_accum_loss_EO += loss_EO.item()
            batch_accum_loss_SAR += loss_SAR.item()
            batch_accum_correct_EO += correct_EO
            batch_accum_correct_SAR += correct_SAR
            batch_accum_total_EO += total_EO
            batch_accum_total_SAR += total_SAR

            # epoch级指标累加（保持不变）
            epoch_train_loss_EO += loss_EO.item()
            epoch_train_loss_SAR += loss_SAR.item()
            epoch_correct_EO += correct_EO
            epoch_correct_SAR += correct_SAR
            epoch_total_EO += total_EO
            epoch_total_SAR += total_SAR

            progress_bar.set_postfix(loss_eo=loss_EO.item(), loss_sar=loss_SAR.item())

            # ========== 核心修改4：每100个batch输出汇总日志 ==========
            if global_batch_count % batch_log_interval == 0:
                # 计算100个batch的平均指标
                avg_loss_EO = batch_accum_loss_EO / batch_log_interval
                avg_loss_SAR = batch_accum_loss_SAR / batch_log_interval
                avg_acc_EO = batch_accum_correct_EO / batch_accum_total_EO if batch_accum_total_EO > 0 else 0.0
                avg_acc_SAR = batch_accum_correct_SAR / batch_accum_total_SAR if batch_accum_total_SAR > 0 else 0.0

                # 输出汇总日志
                print(f"\n===== Global Batch {global_batch_count} (累计{batch_log_interval}个batch) 汇总 =====")
                print(f"平均 Loss_EO: {avg_loss_EO:.4f} | 平均 Accuracy_EO: {100.0 * avg_acc_EO:.2f}%")
                print(f"平均 Loss_SAR: {avg_loss_SAR:.4f} | 平均 Accuracy_SAR: {100.0 * avg_acc_SAR:.2f}%")
                print(f"当前 Epoch: {epoch + 1} | 已处理样本数_EO: {batch_accum_total_EO} | 已处理样本数_SAR: {batch_accum_total_SAR}")
                print("=========================================\n")

                # 重置累计变量
                batch_accum_loss_EO = 0.0
                batch_accum_loss_SAR = 0.0
                batch_accum_correct_EO = 0
                batch_accum_correct_SAR = 0
                batch_accum_total_EO = 0
                batch_accum_total_SAR = 0

        # Epoch级后续处理（保持不变）
        epoch_accuracy_EO = epoch_correct_EO / epoch_total_EO if epoch_total_EO > 0 else 0.0
        epoch_accuracy_SAR = epoch_correct_SAR / epoch_total_SAR if epoch_total_SAR > 0 else 0.0
        scheduler_EO.step(epoch_train_loss_EO)
        scheduler_SAR.step(epoch_train_loss_SAR)

        # Epoch结束时输出简要日志（可选，可注释）
        print(f"\nEpoch {epoch + 1} 结束 | Epoch Loss_EO: {epoch_train_loss_EO/len(train_loader):.4f} | Epoch Acc_EO: {100*epoch_accuracy_EO:.2f}%")
        print(f"Epoch {epoch + 1} 结束 | Epoch Loss_SAR: {epoch_train_loss_SAR/len(train_loader):.4f} | Epoch Acc_SAR: {100*epoch_accuracy_SAR:.2f}%\n")

        # 模型保存（Epoch级）
        h1_EO.remove()
        h1_SAR.remove()
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        torch.save({
            'epoch': epoch + 1,
            'global_batch_count': global_batch_count,
            'model_state_dict': model_SAR.state_dict(),
            'model': model_SAR,
            'optimizer_state_dict': optim_SAR.state_dict(),
            'epoch_loss_SAR': epoch_train_loss_SAR / len(train_loader),
            'epoch_acc_SAR': epoch_accuracy_SAR
        }, os.path.join(CHECKPOINT_DIR, f'SAR_cross_domain_efficientB0_epoch_{epoch + 1}.pth'))
        torch.save({
            'epoch': epoch + 1,
            'global_batch_count': global_batch_count,
            'model_state_dict': model_EO.state_dict(),
            'model': model_EO,
            'optimizer_state_dict': optim_EO.state_dict(),
            'epoch_loss_EO': epoch_train_loss_EO / len(train_loader),
            'epoch_acc_EO': epoch_accuracy_EO
        }, os.path.join(CHECKPOINT_DIR, f'EO_cross_domain_efficientB0_epoch_{epoch + 1}.pth'))

    print('Finished Simultaneous Training')
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(model_EO, os.path.join(CHECKPOINT_DIR, 'EO_cross_domain_efficientB0_final.pth'))
    torch.save(model_SAR, os.path.join(CHECKPOINT_DIR, 'SAR_cross_domain_efficientB0_final.pth'))
    print(f"\n训练完成！总处理batch数: {global_batch_count} | 最终模型已保存至 {CHECKPOINT_DIR}")

if __name__ == "__main__":
    EO_file_pth = os.path.join(BASE_DIR, 'datasets', 'train', 'EO_Train')
    SAR_file_pth = os.path.join(BASE_DIR, 'datasets', 'train', 'SAR_Train')

    if not os.path.isdir(EO_file_pth):
        raise FileNotFoundError(f"训练数据目录不存在: {EO_file_pth}")
    if not os.path.isdir(SAR_file_pth):
        raise FileNotFoundError(f"训练数据目录不存在: {SAR_file_pth}")
    print(f"设备: device_1={device_1}, device_2={device_2} (共 {torch.cuda.device_count()} GPU)")

    transform = transforms.Compose(
        [transforms.Resize(224), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    num_workers = min(4, os.cpu_count() or 1)
    
    # 适配显存的batch_size（48~64之间的最优值）
    batch_size = 60
    train_loader, unlabeled_train_loader, train_dataset_size_EO, train_dataset_size_SAR = prepare_data_loaders(
        EO_file_pth, SAR_file_pth, batch_size=batch_size, test_size=0.2, num_workers=num_workers, 
        eo_transform=transform, sar_transform=transform
    )
    
    # 训练总epoch数（可自定义）
    train(train_loader, unlabeled_train_loader, device_1, device_2, batch_size=batch_size, total_epochs=30) 