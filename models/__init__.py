# from .clearnet import ClearNet, ClearNetResidual  # 이 줄 제거 또는 주석 처리
from .clearnet import ClearNetUnsupervised, DarkChannelPriorLoss, ColorConstancyLoss, TVLoss

def get_model(config):
    model_config = config["model"]
    model_type = model_config["type"]
    
    if model_type == "unet":
        # 이 부분은 ClearNet, ClearNetResidual이 없으므로 에러가 발생할 것입니다
        # 따라서 이 부분은 제거하거나 구현해야 합니다
        raise NotImplementedError("Supervised models (unet) are not implemented yet")
    elif model_type == "unsupervised":
        return ClearNetUnsupervised(
            in_channels=model_config["in_channels"],
            features=model_config["features"],
            use_attention=model_config.get("use_attention", True)
        )
    else:
        raise ValueError(f"Model type {model_type} not supported")


def get_loss_functions(config):
    """손실 함수 생성"""
    if config["model"]["type"] == "unsupervised":
        # 비지도학습용 손실 함수 조합
        losses = {
            "dcp": DarkChannelPriorLoss(
                patch_size=config["loss"].get("dcp_patch_size", 15)
            ),
            "cc": ColorConstancyLoss(),
            "tv": TVLoss(
                weight=config["loss"].get("tv_weight", 1e-6)
            )
        }
        return losses
    else:
        # 기본 지도학습용 손실 함수
        return None