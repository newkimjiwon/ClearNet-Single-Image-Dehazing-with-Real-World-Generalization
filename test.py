import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.utils as vutils
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

# 커스텀 모듈 임포트
from models import get_model


class TestDataset(Dataset):
    """테스트 이미지 데이터셋"""
    def __init__(self, image_dir, img_size=256):
        self.image_dir = image_dir
        self.image_files = sorted([
            f for f in os.listdir(image_dir) 
            if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        # 원본 이미지 크기 저장
        orig_size = image.size
        
        # 텐서로 변환
        image_tensor = self.transform(image)
        
        return {
            "image": image_tensor,
            "filename": self.image_files[idx],
            "orig_size": orig_size
        }


def process_directory(model, dataloader, output_dir, device, save_components=True):
    """디렉토리 내 모든 이미지 처리"""
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    if save_components:
        os.makedirs(os.path.join(output_dir, "transmission"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "atmospheric"), exist_ok=True)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing images"):
            images = batch["image"].to(device)
            filenames = batch["filename"]
            orig_sizes = batch["orig_size"]
            
            # 모델 예측
            clean_images, trans_maps, atmos_lights = model(images)
            
            # 결과 이미지 저장
            for i, (clean, trans, atmos, filename, orig_size) in enumerate(
                zip(clean_images, trans_maps, atmos_lights, filenames, orig_sizes)
            ):
                # 클램핑 및 CPU로 이동
                clean_np = clean.clamp(0, 1).cpu().permute(1, 2, 0).numpy()
                
                # 원본 크기로 리사이즈
                clean_resized = cv2.resize(
                    clean_np, 
                    (orig_size[0], orig_size[1]), 
                    interpolation=cv2.INTER_LINEAR
                )
                
                # 이미지 저장 (0-255 범위로 변환)
                clean_image = (clean_resized * 255).astype(np.uint8)
                clean_image = cv2.cvtColor(clean_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(output_dir, filename), clean_image)
                
                # 컴포넌트 저장 (옵션)
                if save_components:
                    # 투과율 맵 저장
                    trans_np = trans.repeat(3, 1, 1).clamp(0, 1).cpu().permute(1, 2, 0).numpy()
                    trans_resized = cv2.resize(
                        trans_np, 
                        (orig_size[0], orig_size[1]), 
                        interpolation=cv2.INTER_LINEAR
                    )
                    trans_image = (trans_resized * 255).astype(np.uint8)
                    trans_image = cv2.cvtColor(trans_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(output_dir, "transmission", filename), trans_image)
                    
                    # 대기광 저장
                    atmos_np = atmos.clamp(0, 1).cpu().permute(1, 2, 0).numpy()
                    atmos_resized = cv2.resize(
                        atmos_np, 
                        (orig_size[0], orig_size[1]), 
                        interpolation=cv2.INTER_LINEAR
                    )
                    atmos_image = (atmos_resized * 255).astype(np.uint8)
                    atmos_image = cv2.cvtColor(atmos_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(output_dir, "atmospheric", filename), atmos_image)


def process_single_image(model, image_path, output_dir, device, img_size=256):
    """단일 이미지 처리"""
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    # 이미지 로드
    image = Image.open(image_path).convert('RGB')
    filename = os.path.basename(image_path)
    orig_size = image.size
    
    # 전처리
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # 모델 예측
        clean_image, trans_map, atmos_light = model(image_tensor)
        
        # 결과 시각화
        plt.figure(figsize=(15, 10))
        
        # 원본 이미지
        plt.subplot(2, 2, 1)
        plt.title("Input Hazy Image")
        plt.imshow(np.array(image))
        plt.axis('off')
        
        # 복원된 이미지
        plt.subplot(2, 2, 2)
        clean_np = clean_image[0].clamp(0, 1).cpu().permute(1, 2, 0).numpy()
        plt.title("Dehazed Image")
        plt.imshow(clean_np)
        plt.axis('off')
        
        # 투과율 맵
        plt.subplot(2, 2, 3)
        trans_np = trans_map[0].repeat(3, 1, 1).clamp(0, 1).cpu().permute(1, 2, 0).numpy()
        plt.title("Transmission Map")
        plt.imshow(trans_np)
        plt.axis('off')
        
        # 대기광
        plt.subplot(2, 2, 4)
        atmos_np = atmos_light[0].clamp(0, 1).cpu().permute(1, 2, 0).numpy()
        plt.title("Atmospheric Light")
        plt.imshow(atmos_np)
        plt.axis('off')
        
        # 결과 저장
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"analysis_{filename}"))
        plt.close()
        
        # 원본 크기로 복원된 이미지 저장
        clean_np = clean_image[0].clamp(0, 1).cpu().permute(1, 2, 0).numpy()
        clean_resized = cv2.resize(
            clean_np, 
            (orig_size[0], orig_size[1]), 
            interpolation=cv2.INTER_LINEAR
        )
        clean_image_pil = Image.fromarray((clean_resized * 255).astype(np.uint8))
        clean_image_pil.save(os.path.join(output_dir, f"dehazed_{filename}"))


def main():
    # 인자 파싱
    parser = argparse.ArgumentParser(description="Test unsupervised dehazing model")
    parser.add_argument("--config", type=str, default="config_unsupervised.json", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Path to input image or directory")
    parser.add_argument("--output", type=str, default="./path/output/results", help="Path to output directory")
    parser.add_argument("--save_components", action="store_true", help="Save transmission map and atmospheric light")
    args = parser.parse_args()
    
    # 설정 로드
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # 디바이스 설정
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 모델 로드
    model = get_model(config).to(device)
    
    # 체크포인트 로드
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    print(f"Model loaded from {args.checkpoint}")
    
    # 입력이 디렉토리인지 파일인지 확인
    if os.path.isdir(args.input):
        # 디렉토리 처리
        test_dataset = TestDataset(
            args.input, 
            img_size=config["data"]["img_size"]
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4
        )
        print(f"Processing directory with {len(test_dataset)} images...")
        process_directory(model, test_dataloader, args.output, device, args.save_components)
    else:
        # 단일 이미지 처리
        print(f"Processing single image: {args.input}")
        process_single_image(
            model, 
            args.input, 
            args.output, 
            device, 
            img_size=config["data"]["img_size"]
        )
    
    print(f"Processing complete. Results saved to {args.output}")


if __name__ == "__main__":
    main()