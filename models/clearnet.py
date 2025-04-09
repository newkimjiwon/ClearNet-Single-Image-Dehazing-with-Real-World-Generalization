import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi


class HazeRemovalNet(nn.Module):
    """안개 제거 네트워크 - 안개 맵과 투과율 예측"""
    def __init__(self, in_channels=3, features=[64, 128, 256, 512], use_attention=True):
        super(HazeRemovalNet, self).__init__()
        self.use_attention = use_attention
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Down part
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
            
        # Up part
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))
            
        # Attention gates
        if self.use_attention:
            self.attention_gates = nn.ModuleList()
            for feature in reversed(features):
                self.attention_gates.append(AttentionGate(feature, feature, feature//2))
            
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        
        # 투과율 맵(transmission map)과 대기광(atmospheric light) 예측을 위한 출력 레이어
        self.final_trans = nn.Conv2d(features[0], 1, kernel_size=1)  # 투과율 맵 (0~1)
        self.final_atmos = nn.Conv2d(features[0], 3, kernel_size=1)  # 대기광 (RGB)
        
    def forward(self, x):
        skip_connections = []
        
        # Encoder path
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
            
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # Reverse list
        
        # Decoder path
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            
            # Handle different sizes
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode="bilinear", align_corners=True)
                
            if self.use_attention:
                skip_connection = self.attention_gates[idx//2](x, skip_connection)
                
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        
        # 투과율 맵 예측 (0~1 범위)
        trans_map = torch.sigmoid(self.final_trans(x))
        
        # 대기광 예측 (0~1 범위)
        atmos_light = torch.sigmoid(self.final_atmos(x))
        
        return trans_map, atmos_light


class ClearNetUnsupervised(nn.Module):
    """비지도학습 기반 안개 제거 모델"""
    def __init__(self, in_channels=3, features=[64, 128, 256, 512], use_attention=True):
        super(ClearNetUnsupervised, self).__init__()
        self.haze_net = HazeRemovalNet(in_channels, features, use_attention)
        
    def forward(self, x):
        # 투과율 맵과 대기광 예측
        trans_map, atmos_light = self.haze_net(x)
        
        # 대기 산란 모델(Atmospheric Scattering Model)을 이용한 안개 제거
        # J = (I - A) / t + A, 여기서 J는 깨끗한 이미지, I는 안개 이미지, A는 대기광, t는 투과율
        # 안전성을 위해 투과율의 최소값 제한
        trans_map_safe = torch.clamp(trans_map, min=0.1)
        
        # 각 채널에 대해 대기광 처리
        clean_image = torch.zeros_like(x)
        for c in range(3):
            clean_image[:, c:c+1, :, :] = (x[:, c:c+1, :, :] - atmos_light[:, c:c+1, :, :]) / trans_map_safe + atmos_light[:, c:c+1, :, :]
        
        # 결과 이미지 범위 조정 (0~1)
        clean_image = torch.clamp(clean_image, 0, 1)
        
        return clean_image, trans_map, atmos_light


class DarkChannelPriorLoss(nn.Module):
    """Dark Channel Prior 기반 손실 함수"""
    def __init__(self, patch_size=15, eps=1e-6):
        super(DarkChannelPriorLoss, self).__init__()
        self.patch_size = patch_size
        self.eps = eps
        self.pool = nn.MaxPool2d(kernel_size=patch_size, stride=1, padding=patch_size//2)
    
    def forward(self, clean_image):
        # 각 채널에서 최솟값 계산
        dark_channel = torch.min(clean_image, dim=1, keepdim=True)[0]
        
        # 패치 내에서 최댓값 계산 (= 패치 내 최소값의 최대값)
        # 이것은 dark channel prior를 위배하는 정도를 측정
        dark_channel_pooled = -self.pool(-dark_channel)
        
        # 깨끗한 이미지의 dark channel은 0에 가까워야 함
        # 손실 = dark channel 값의 제곱 평균
        loss = torch.mean(dark_channel_pooled ** 2)
        return loss


class ColorConstancyLoss(nn.Module):
    """색상 일관성 손실"""
    def __init__(self):
        super(ColorConstancyLoss, self).__init__()
    
    def forward(self, clean_image):
        # 각 채널의 평균을 계산
        mean_rgb = torch.mean(clean_image, dim=[2, 3], keepdim=True)
        
        # 채널 간 편차 계산 (회색 세계 가정 기반)
        r, g, b = mean_rgb[:, 0:1, :, :], mean_rgb[:, 1:2, :, :], mean_rgb[:, 2:3, :, :]
        loss = ((r - g)**2 + (r - b)**2 + (g - b)**2) / 3
        return torch.mean(loss)


class TVLoss(nn.Module):
    """Total Variation Loss - 이미지 부드러움 유지"""
    def __init__(self, weight=1):
        super(TVLoss, self).__init__()
        self.weight = weight
        
    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x-1]), 2).sum()
        return self.weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size
    
    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


def test():
    """모델 테스트 함수"""
    # 입력 예시
    x = torch.randn((2, 3, 256, 256))
    
    # 모델 생성
    model = ClearNetUnsupervised(in_channels=3)
    
    # 추론
    clean_image, trans_map, atmos_light = model(x)
    
    # 결과 출력
    print(f"Input shape: {x.shape}")
    print(f"Clean image shape: {clean_image.shape}")
    print(f"Transmission map shape: {trans_map.shape}")
    print(f"Atmospheric light shape: {atmos_light.shape}")
    
    # 총 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # 손실 함수 테스트
    dcp_loss = DarkChannelPriorLoss()
    cc_loss = ColorConstancyLoss()
    tv_loss = TVLoss()
    
    loss_dcp = dcp_loss(clean_image)
    loss_cc = cc_loss(clean_image)
    loss_tv = tv_loss(clean_image)
    
    print(f"DCP Loss: {loss_dcp.item()}")
    print(f"Color Constancy Loss: {loss_cc.item()}")
    print(f"TV Loss: {loss_tv.item()}")


if __name__ == "__main__":
    test()