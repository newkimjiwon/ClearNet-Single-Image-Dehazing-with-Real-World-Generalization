import os
import json
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
import cv2
from sklearn.model_selection import train_test_split
import shutil


def create_directory(directory):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


def preprocess_unsupervised_dataset(config, dataset_name):
    """
    비지도학습을 위한 데이터셋 전처리
    
    비지도학습의 특징:
    - 깨끗한 이미지(Ground Truth) 없이 안개 이미지만으로 학습
    - 이미지 특성과 물리적 모델(대기 산란 모델)을 활용한 자체 감독 학습
    - 다양한 안개 밀도와 조건에 적응할 수 있는 전처리 적용
    
    전처리 단계:
    1. 이미지 크기 표준화
    2. 학습/검증 분할 (70%/30%)
    3. 데이터 강화를 위한 이미지 변환:
       - 히스토그램 분석으로 안개 특성 추출
       - 안개 강도 다양화를 위한 이미지 변형
    """
    print(f"Preprocessing {dataset_name} dataset for unsupervised learning...")
    
    if dataset_name == "RESIDE":
        hazy_path = config["data"].get("hazy_train_path", config["data"]["train_data_path"])
    elif dataset_name == "NH-HAZE":
        hazy_path = config["data"]["real_test_path"]
    else:
        hazy_path = config["data"].get(f"{dataset_name.lower()}_path", "")
        if not hazy_path:
            print(f"Path for {dataset_name} not found in config")
            return

    output_path = os.path.join("./data_set", f"{dataset_name}_unsupervised")
    img_size = config["data"]["img_size"]
    
    # 학습/검증 비율 설정
    train_ratio = 0.7
    
    # 출력 디렉토리 생성
    create_directory(output_path)
    create_directory(os.path.join(output_path, "train"))
    create_directory(os.path.join(output_path, "val"))
    
    # 파일 경로 읽기
    hazy_files = sorted([f for f in os.listdir(hazy_path) if f.endswith(('.png', '.jpg', '.jpeg'))])

    # 파일이 없으면 건너뛰기
    if len(hazy_files) == 0:
        print(f"No images found in {hazy_path}. Skipping {dataset_name} dataset.")
        return
    
    # 학습/검증 분할
    train_files, val_files = train_test_split(
        hazy_files, 
        test_size=1-train_ratio, 
        random_state=42
    )
    
    print(f"Found {len(hazy_files)} images, split into {len(train_files)} train and {len(val_files)} validation")
    
    # 기본 변환 - 크기 조정 및 텐서 변환
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    
    # 안개 이미지 변형을 위한 함수
    def apply_haze_augmentation(img, intensity_factor=None):
        """
        안개 특성을 변형하여 다양한 학습 데이터 생성
        intensity_factor: None이면 랜덤 값, 아니면 지정된 값으로 안개 강도 조절
        """
        img_cv = np.array(img)
        
        # HSV 색 공간으로 변환
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2HSV)
        
        # 안개 강도 조절 (랜덤 또는 지정값)
        if intensity_factor is None:
            # 0.7~1.3 사이 랜덤한 강도 계수
            intensity_factor = random.uniform(0.7, 1.3)
        
        # 값(V) 채널 조절 - 안개 밝기 효과
        hsv[:,:,2] = np.clip(hsv[:,:,2] * intensity_factor, 0, 255).astype('uint8')
        
        # 채도(S) 채널 조절 - 안개 농도 효과 
        saturation_factor = 1.0 / intensity_factor  # 안개가 짙을수록 채도는 낮아짐
        hsv[:,:,1] = np.clip(hsv[:,:,1] * saturation_factor, 0, 255).astype('uint8')
        
        # RGB로 변환
        augmented = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return Image.fromarray(augmented), intensity_factor
    
    # 학습 데이터 처리
    print(f"Processing {dataset_name} training data...")
    for idx, hazy_file in enumerate(tqdm(train_files)):
        # 이미지 로드
        hazy_img = Image.open(os.path.join(hazy_path, hazy_file)).convert('RGB')
        
        # 원본 안개 이미지 저장
        hazy_tensor = transform(hazy_img)
        torch.save(hazy_tensor, os.path.join(output_path, "train", f"original_{idx:04d}.pt"))
        
        # 3개의 추가 안개 변형 생성 (다양한 안개 강도)
        for aug_idx in range(3):
            # 안개 강도를 다양하게 변형
            augmented_img, factor = apply_haze_augmentation(hazy_img)
            
            # 텐서로 변환 및 저장
            aug_tensor = transform(augmented_img)
            torch.save(aug_tensor, os.path.join(output_path, "train", f"aug{aug_idx}_{idx:04d}_f{factor:.2f}.pt"))
    
    # 검증 데이터 처리
    print(f"Processing {dataset_name} validation data...")
    for idx, hazy_file in enumerate(tqdm(val_files)):
        # 이미지 로드 - 검증 데이터는 원본만 사용
        hazy_img = Image.open(os.path.join(hazy_path, hazy_file)).convert('RGB')
        
        # 텐서로 변환 및 저장
        hazy_tensor = transform(hazy_img)
        torch.save(hazy_tensor, os.path.join(output_path, "val", f"{idx:04d}.pt"))
    
    print(f"Preprocessing complete. Saved to {output_path}")
    print(f"Train/Val split: {len(train_files)}(+{len(train_files)*3} augmented)/{len(val_files)} images")


def main():
    # 인자 파싱
    parser = argparse.ArgumentParser(description="Preprocess datasets for unsupervised dehazing")
    parser.add_argument("--config", type=str, default="config_unsupervised.json", help="Path to config file")
    parser.add_argument("--dataset", type=str, default="all", 
                        choices=["all", "RESIDE", "NH-HAZE"], 
                        help="Dataset to preprocess")
    args = parser.parse_args()
    
    # 설정 로드
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # 기본 디렉토리 생성
    create_directory("./data_set")
    
    # 선택된 데이터셋 처리
    if args.dataset == "all" or args.dataset == "RESIDE":
        preprocess_unsupervised_dataset(config, "RESIDE")
    
    if args.dataset == "all" or args.dataset == "NH-HAZE":
        preprocess_unsupervised_dataset(config, "NH-HAZE")
    
    print("All preprocessing complete!")


if __name__ == "__main__":
    main()