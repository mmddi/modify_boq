#!/usr/bin/env python3
import torch
from src.backbones import DinoV2
from src.boq import BoQ

def test_without_adapter():
    """测试纯BoQ（无Adapter）的输出"""
    print("="*70)
    print("测试纯BoQ输出（无Adapter，无预训练权重）")
    print("="*70)
    
    backbone = DinoV2(backbone_name="dinov2_vitb14", pretrained=True, unfreeze_n_blocks=2)
    
    aggregator = BoQ(in_channels=768, proj_channels=384, num_queries=64, num_layers=2, row_dim=32)
    
    # 随机初始化（无预训练）
    print("\n[1] 随机初始化BoQ:")
    backbone.eval()
    aggregator.eval()
    with torch.no_grad():
        dummy = torch.randn(2, 3, 224, 224)
        features = backbone(dummy)
        desc, _ = aggregator(features)
        desc = torch.nn.functional.normalize(desc, p=2, dim=-1)
        print(f"  desc shape: {desc.shape}")
        print(f"  mean: {desc.mean().item():.4f}")
        print(f"  std: {desc.std().item():.4f}")
        print(f"  max: {desc.max().item():.4f}")
        print(f"  min: {desc.min().item():.4f}")
    
    # 加载预训练权重
    print("\n[2] 加载预训练BoQ权重:")
    state_dict = torch.load("src/dinov2_12288.pth", map_location='cpu')
    agg_state = {k[len('aggregator.'):]: v for k, v in state_dict.items() if k.startswith('aggregator.')}
    aggregator.load_state_dict(agg_state, strict=False)
    
    with torch.no_grad():
        desc, _ = aggregator(features)  # 复用之前的features
        desc = torch.nn.functional.normalize(desc, p=2, dim=-1)
        print(f"  desc shape: {desc.shape}")
        print(f"  mean: {desc.mean().item():.4f}")
        print(f"  std: {desc.std().item():.4f}")
        print(f"  max: {desc.max().item():.4f}")
        print(f"  min: {desc.min().item():.4f}")
        
    with torch.no_grad():
	    desc_raw, _ = aggregator(features)
	    # 不归一化！
	    print(f"  RAW desc (no normalize):")
	    print(f"    mean: {desc_raw.mean().item():.4f}")
	    print(f"    std: {desc_raw.std().item():.4f}")
	    print(f"    max: {desc_raw.max().item():.4f}")
	    print(f"    min: {desc_raw.min().item():.4f}")
	    
	    # 归一化后
	    desc_norm = torch.nn.functional.normalize(desc_raw, p=2, dim=-1)
	    print(f"  NORMALIZED desc:")
	    print(f"    mean: {desc_norm.mean().item():.4f}")
	    print(f"    std: {desc_norm.std().item():.4f}")
    
    print("="*70)

if __name__ == "__main__":
    test_without_adapter()
