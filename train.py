import os
import json
import argparse
import numpy as np
from tqdm import tqdm
import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torchvision import transforms
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter

from PIL import Image
import matplotlib.pyplot as plt

# 커스텀 모듈 임포트
from models import get_model, get_loss_functions
from utils.metrics import psnr, ssim, batch_psnr, SSIM


# 재현성을 위한 시드 설정
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class UnsupervisedDehazingDataset(Dataset):
    """비지도학습용 안개 제거 데이터셋"""
    def __init__(self, hazy_dir, augment=False):
        """
        Args:
            hazy_dir (str): 안개 이미지가 있는 디렉토리 경로
            augment (bool): 데이터 증강 사용 여부
        """
        self.hazy_dir = hazy_dir
        self.augment = augment
        
        # PT 파일 목록 가져오기
        self.hazy_files = sorted([f for f in os.listdir(hazy_dir) if f.endswith('.pt')])
        
        # 데이터 증강 변환
        self.transform_augment = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1)  # 밝기만 약간 변경
        ])
    
    def __len__(self):
        return len(self.hazy_files)
    
    def __getitem__(self, idx):
        # 이미지 파일 경로
        hazy_path = os.path.join(self.hazy_dir, self.hazy_files[idx])
        
        # 저장된 텐서 로드
        hazy_tensor = torch.load(hazy_path)
        
        # 데이터 증강 적용 (필요시)
        if self.augment:
            # PIL 이미지로 변환하여 증강 적용
            hazy_pil = transforms.ToPILImage()(hazy_tensor)
            hazy_aug = self.transform_augment(hazy_pil)
            
            # 다시 텐서로 변환
            hazy_tensor = transforms.ToTensor()(hazy_aug)
        
        return {
            "hazy": hazy_tensor,
            "filename": self.hazy_files[idx]
        }


def train_one_epoch(model, dataloader, optimizer, loss_functions, loss_weights, device, epoch):
    """한 에폭 학습"""
    model.train()
    epoch_loss = 0
    epoch_metrics = {name: 0 for name in loss_functions.keys()}
    
    loop = tqdm(dataloader, leave=True)
    for batch_idx, batch in enumerate(loop):
        # 데이터 디바이스로 이동
        hazy_images = batch["hazy"].to(device)
        
        # 모델 예측
        clean_images, trans_maps, atmos_lights = model(hazy_images)
        
        # 각 손실 함수 계산
        losses = {}
        for name, loss_fn in loss_functions.items():
            if name == "dcp":
                losses[name] = loss_fn(clean_images) * loss_weights[name]
            elif name == "cc":
                losses[name] = loss_fn(clean_images) * loss_weights[name]
            elif name == "tv":
                # Transmission map과 atmospheric light에 TV 손실 적용
                losses[name] = (loss_fn(trans_maps) + loss_fn(atmos_lights)) * loss_weights[name]
                
        # 총 손실 계산
        total_loss = sum(losses.values())
        
        # 역전파 및 최적화
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # 손실 누적
        epoch_loss += total_loss.item()
        for name, loss in losses.items():
            epoch_metrics[name] += loss.item()
            
        # 진행 상황 업데이트
        loop.set_description(f"Epoch [{epoch+1}]")
        loop.set_postfix(loss=total_loss.item())
    
    # 평균 손실 및 메트릭 계산
    epoch_loss /= len(dataloader)
    for name in epoch_metrics:
        epoch_metrics[name] /= len(dataloader)
    
    return epoch_loss, epoch_metrics


def validate(model, dataloader, loss_functions, loss_weights, device):
    """검증 데이터에 대한 평가"""
    model.eval()
    val_loss = 0
    val_metrics = {name: 0 for name in loss_functions.keys()}
    
    with torch.no_grad():
        for batch in dataloader:
            # 데이터 디바이스로 이동
            hazy_images = batch["hazy"].to(device)
            
            # 모델 예측
            clean_images, trans_maps, atmos_lights = model(hazy_images)
            
            # 각 손실 함수 계산
            losses = {}
            for name, loss_fn in loss_functions.items():
                if name == "dcp":
                    losses[name] = loss_fn(clean_images) * loss_weights[name]
                elif name == "cc":
                    losses[name] = loss_fn(clean_images) * loss_weights[name]
                elif name == "tv":
                    losses[name] = (loss_fn(trans_maps) + loss_fn(atmos_lights)) * loss_weights[name]
                    
            # 총 손실 계산
            total_loss = sum(losses.values())
            
            # 손실 누적
            val_loss += total_loss.item()
            for name, loss in losses.items():
                val_metrics[name] += loss.item()
    
    # 평균 손실 및 메트릭 계산
    val_loss /= len(dataloader)
    for name in val_metrics:
        val_metrics[name] /= len(dataloader)
        
    return val_loss, val_metrics


def save_checkpoint(model, optimizer, scheduler, epoch, loss, config, filename):
    """체크포인트 저장"""
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "loss": loss,
        "config": config
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")


def save_samples(model, dataloader, device, output_dir, epoch, num_samples=4):
    """샘플 결과 이미지 저장"""
    model.eval()
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        # 첫 배치에서 샘플 선택
        for batch in dataloader:
            hazy_images = batch["hazy"][:num_samples].to(device)
            filenames = batch["filename"][:num_samples]
            
            # 모델 예측
            clean_images, trans_maps, atmos_lights = model(hazy_images)
            
            # 각 샘플 저장
            for i in range(len(hazy_images)):
                # 결과 시각화를 위한 이미지 그리드 생성
                grid = torch.cat([
                    hazy_images[i].cpu(),            # 원본 안개 이미지
                    clean_images[i].cpu(),           # 복원된 깨끗한 이미지
                    trans_maps[i].repeat(3, 1, 1).cpu(),  # 투과율 맵 (시각화를 위해 3채널로 복제)
                    atmos_lights[i].cpu()            # 대기광
                ], dim=2)
                
                # 이미지 저장
                vutils.save_image(
                    grid, 
                    os.path.join(output_dir, f"epoch_{epoch+1}_{filenames[i].split('.')[0]}.png"),
                    normalize=True
                )
            
            # 첫 배치만 처리
            break


def main():
    # 인자 파싱
    parser = argparse.ArgumentParser(description="Train unsupervised dehazing model")
    parser.add_argument("--config", type=str, default="config_unsupervised.json", help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint for resuming training")
    args = parser.parse_args()
    
    # 설정 로드
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # 디렉토리 생성
    os.makedirs(config["paths"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["paths"]["output_dir"], exist_ok=True)
    os.makedirs(config["paths"]["log_dir"], exist_ok=True)
    
    # 재현성을 위한 시드 설정
    set_seed()
    
    # 디바이스 설정
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 데이터셋 및 데이터로더 생성
    train_dataset = UnsupervisedDehazingDataset(
        os.path.join("./data_set", "RESIDE_unsupervised", "train"),
        augment=config["data"]["use_augmentation"]
    )
    
    val_dataset = UnsupervisedDehazingDataset(
        os.path.join("./data_set", "RESIDE_unsupervised", "val"),
        augment=False
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train dataset: {len(train_dataset)} images")
    print(f"Validation dataset: {len(val_dataset)} images")
    
    # 모델 생성
    model = get_model(config).to(device)
    print(f"Model: {config['model']['name']}")
    
    # 손실 함수 생성
    loss_functions = get_loss_functions(config)
    
    # 손실 가중치 설정
    loss_weights = {
        "dcp": config["loss"]["dcp_weight"],
        "cc": config["loss"]["cc_weight"],
        "tv": config["loss"]["tv_weight"]
    }
    
    # 옵티마이저 및 스케줄러 설정
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )
    
    scheduler = StepLR(
        optimizer,
        step_size=config["training"]["scheduler_step_size"],
        gamma=config["training"]["scheduler_gamma"]
    )
    
    # TensorBoard 로그 작성기 생성
    log_dir = os.path.join(
        config["paths"]["log_dir"],
        datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard log directory: {log_dir}")
    
    # 학습 시작
    start_epoch = 0
    best_loss = float('inf')
    
    # 체크포인트에서 학습 재개 (필요시)
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            if checkpoint["scheduler"]:
                scheduler.load_state_dict(checkpoint["scheduler"])
            start_epoch = checkpoint["epoch"] + 1
            best_loss = checkpoint["loss"]
            print(f"Resuming from epoch {start_epoch}, best loss: {best_loss:.6f}")
        else:
            print(f"No checkpoint found at {args.resume}")
    
    # 학습 루프
    num_epochs = config["training"]["num_epochs"]
    for epoch in range(start_epoch, num_epochs):
        # 학습
        train_loss, train_metrics = train_one_epoch(
            model, train_dataloader, optimizer, loss_functions, loss_weights, device, epoch
        )
        
        # 검증
        val_loss, val_metrics = validate(
            model, val_dataloader, loss_functions, loss_weights, device
        )
        
        # 학습률 스케줄러 업데이트
        scheduler.step()
        
        # 현재 학습률 로깅
        current_lr = scheduler.get_last_lr()[0]
        
        # TensorBoard에 결과 기록
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("LR", current_lr, epoch)
        
        for name, value in train_metrics.items():
            writer.add_scalar(f"Metrics/{name}_train", value, epoch)
        
        for name, value in val_metrics.items():
            writer.add_scalar(f"Metrics/{name}_val", value, epoch)
        
        # 샘플 결과 이미지 저장
        if (epoch + 1) % config["training"]["save_interval"] == 0:
            save_samples(
                model,
                val_dataloader,
                device,
                os.path.join(config["paths"]["output_dir"], "samples"),
                epoch
            )
        
        # 최고 성능 모델 체크포인트 저장
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                best_loss,
                config,
                os.path.join(config["paths"]["checkpoint_dir"], "best_model.pth")
            )
        
        # 중간 체크포인트 저장
        if (epoch + 1) % config["training"]["save_interval"] == 0:
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                val_loss,
                config,
                os.path.join(config["paths"]["checkpoint_dir"], f"checkpoint_epoch_{epoch+1}.pth")
            )
        
        # 진행 상황 출력
        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
              f"LR: {current_lr:.8f}")
    
    # 최종 모델 저장
    save_checkpoint(
        model,
        optimizer,
        scheduler,
        num_epochs - 1,
        val_loss,
        config,
        os.path.join(config["paths"]["checkpoint_dir"], "final_model.pth")
    )
    
    # TensorBoard 작성기 닫기
    writer.close()
    
    print("Training complete!")


if __name__ == "__main__":
    main()