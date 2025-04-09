# ClearNet-Single-Image-Dehazing-with-Real-World-Generalization

이 프로젝트는 딥러닝을 활용한 단일 이미지 안개 제거 모델 'ClearNet'을 구현합니다. 비지도학습 접근법을 사용하여 안개가 있는 이미지에서 안개를 제거하고 원본 이미지를 복원합니다.

## 개요

**ClearNet**은 안개 낀 이미지에서 물리적 안개 산란 모델(Atmospheric Scattering Model)을 학습하여 안개를 효과적으로 제거합니다. 깨끗한 참조 이미지 없이도 학습이 가능한 비지도학습 방식을 채택하여 실용성을 높였습니다.

### 주요 특징

- **비지도학습 접근법**: 안개가 있는 이미지만으로 학습 가능
- **물리 기반 모델링**: 대기 산란 모델을 기반으로 투과율 맵과 대기광 추정
- **어텐션 메커니즘**: 상세 정보 보존을 위한 어텐션 게이트 적용
- **실시간 처리 가능**: 효율적인 U-Net 기반 아키텍처

### 응용 분야

- CCTV 영상 복원 (범죄 수사)
- 자율주행 시야 개선
- 드론 영상 복원
- 블랙박스 사고 분석
- 사진 및 영상 품질 개선

## 설치 및 환경 설정

### 요구 사항

- Python 3.9+
- PyTorch 1.12.1
- CUDA 11.3
- NVIDIA RTX 3090 GPU (권장)

### 설치 방법

1. 환경 설정:
```bash
conda create -n clearnet python=3.9
conda activate clearnet
```

2. 필수 라이브러리 설치:
```bash
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install numpy==1.23.5 matplotlib opencv-python scikit-image scikit-learn tqdm tensorboard pyyaml 
```

## 프로젝트 구조

```
ClearNet/
├── data/               # 원본 데이터셋
│   └── hazy/           # 안개 이미지
├── data_set/           # 전처리된 데이터
│   └── RESIDE_unsupervised/
│       ├── train/      # 학습 데이터 (70%)
│       └── val/        # 검증 데이터 (30%)
├── models/             # 모델 정의
│   ├── __init__.py     # 모델 초기화
│   ├── clearnet.py     # 모델 아키텍처
│   └── checkpoints/    # 학습된 모델 저장
├── utils/              # 유틸리티 함수
│   ├── __init__.py
│   ├── metrics.py      # 평가 지표 (PSNR, SSIM)
│   └── logs/           # TensorBoard 로그
├── path/
│   ├── input/          # 테스트 이미지
│   └── output/         # 결과 이미지
├── pre_process.py      # 데이터 전처리
├── train.py            # 모델 학습
├── test.py             # 모델 테스트
└── config.json         # 설정 파일
```

## 사용 방법

### 1. 데이터 전처리

모델 학습을 위한 데이터 전처리:

```bash
python pre_process.py --config config.json --dataset RESIDE
```

### 2. 모델 학습

특정 GPU(예: 2번)를 사용한 학습:

```bash
CUDA_VISIBLE_DEVICES=2 python train.py --config config.json
```

학습 진행 상황은 TensorBoard를 통해 모니터링 가능:

```bash
tensorboard --logdir=./utils/logs
```

### 3. 모델 테스트

학습된 모델을 사용한 안개 제거 테스트:

```bash
python test.py --config config.json --checkpoint ./models/checkpoints/best_model.pth --input ./path/input/ --output ./path/output/
```

단일 이미지 테스트:

```bash
python test.py --config config.json --checkpoint ./models/checkpoints/best_model.pth --input ./path/input/hazy_image.jpg --output ./path/output/
```

## 모델 아키텍처

ClearNet은 다음과 같은 요소로 구성됩니다:

1. **안개 제거 네트워크 (HazeRemovalNet)**: 
   - U-Net 기반 인코더-디코더 구조
   - 어텐션 게이트로 상세 정보 보존
   - 투과율 맵과 대기광 예측

2. **손실 함수**:
   - **DarkChannelPriorLoss**: 자연 이미지의 통계적 특성 활용
   - **ColorConstancyLoss**: 색상 정보 보존
   - **TVLoss**: 이미지 부드러움 유지

## 성능 및 결과

모델은 안개 이미지에서 안개를 효과적으로 제거하면서 이미지 품질을 개선합니다. 결과는 `path/output/samples` 폴더에서 확인할 수 있으며, 다음과 같은 형식으로 저장됩니다:

- 원본 안개 이미지
- 안개가 제거된 이미지
- 투과율 맵
- 추정된 대기광

## 주의사항

- 색상 보존 문제: 현재 모델은 일부 이미지에서 색상 정보 손실이 발생할 수 있습니다. 이는 손실 함수 가중치 조정으로 개선 가능합니다.
- 메모리 사용량: 고해상도 이미지 처리 시 GPU 메모리 사용량에 주의하세요.

## 향후 개선 사항

- 색상 보존을 위한 손실 함수 최적화
- 실시간 처리를 위한 모델 경량화
- 다양한 날씨 조건(안개, 비, 눈 등)에 대한 확장
