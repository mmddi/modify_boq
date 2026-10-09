#!/usr/bin/env python3
import torch
from src.backbones import DinoV2
from src.boq import BoQ
from src.dino_boq_view_adapter import DinoBoQViewSpecificAdapterEncoder

def verify():
    print("="*70)
    print("验证预训练BoQ权重加载效果")
    print("="*70)
    
    # 1. 构建模型（与训练时相同配置）
    backbone = DinoV2(
        backbone_name="dinov2_vitb14",
        pretrained=True,
        unfreeze_n_blocks=2,
    )
    
    aggregator = BoQ(
        in_channels=768,
        proj_channels=384,
        num_queries=64,
        num_layers=2,
        row_dim=32,  # 12288 // 384
    )
    
    image_encoder = DinoBoQViewSpecificAdapterEncoder(
        backbone=backbone,
        aggregator=aggregator,
        normalize_descriptor=True,
        use_view_specific_adapter=True,
        adapter_bottleneck_dim=128,
        adapter_scale=0.1,
        adapter_zero_init=True,
        freeze_backbone=False,
        train_ground_adapter=True,
        train_sat_adapter=True,
    )
    
    # 2. 加载前测试
    print("\n[1] 加载权重前的描述子统计：")
    image_encoder.eval()
    with torch.no_grad():
        dummy = torch.randn(2, 3, 224, 224)
        desc_before = image_encoder(dummy, view_type="ground")
        print(f"  shape: {desc_before.shape}")
        print(f"  mean: {desc_before.mean().item():.4f}")
        print(f"  std: {desc_before.std().item():.4f}")
        print(f"  max: {desc_before.max().item():.4f}")
        print(f"  min: {desc_before.min().item():.4f}")
    
    # 3. 加载权重
    print("\n[2] 加载预训练权重...")
    from src.pretrained_boq import load_pretrained_boq_into_view_adapter_encoder
    
    result = load_pretrained_boq_into_view_adapter_encoder(
        image_encoder=image_encoder,
        backbone_name="dinov2",
        output_dim=12288,
        map_location="cpu",
        strict=False,
        checkpoint_path="src/dinov2_12288.pth",
    )
    
    print(f"  Backbone loaded: {result['shared_backbone']['loaded_keys']} keys")
    print(f"  Aggregator loaded: {result['shared_aggregator']['loaded_keys']} keys")
    
    # 4. 加载后测试
    print("\n[3] 加载权重后的描述子统计：")
    with torch.no_grad():
        desc_after = image_encoder(dummy, view_type="ground")
        print(f"  shape: {desc_after.shape}")
        print(f"  mean: {desc_after.mean().item():.4f}")
        print(f"  std: {desc_after.std().item():.4f}")
        print(f"  max: {desc_after.max().item():.4f}")
        print(f"  min: {desc_after.min().item():.4f}")
    
    # 5. 判断
    print("\n[4] 诊断结果：")
    std_before = desc_before.std().item()
    std_after = desc_after.std().item()
    
    if std_after < 0.05:
        print("  ❌ 异常：加载后 std < 0.05，预训练权重可能未生效")
        print("  可能原因：")
        print("    - 权重文件损坏或不完整")
        print("    - 权重键名与模型不匹配")
        print("    - 权重被后续操作覆盖")
    else:
        print("  ✅ 正常：加载后 std > 0.05，预训练权重已生效")
    
    if std_after <= std_before:
        print("  ❌ 异常：加载后 std 未提升，权重可能未正确加载")
    else:
        print(f"  ✅ 正常：std 从 {std_before:.4f} 提升到 {std_after:.4f}")
    
    print("="*70)

if __name__ == "__main__":
    verify()
