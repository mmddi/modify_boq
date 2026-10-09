
import torch
import numpy as np

def deep_verify():
    print("="*70)
    print("深度诊断：检查预训练权重文件内容")
    print("="*70)
    
    # 1. 直接加载权重文件
    checkpoint_path = "src/dinov2_12288.pth"
    print(f"\\n[1] 加载权重文件: {checkpoint_path}")
    
    try:
        state_dict = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return
    
    print(f"    总键数: {len(state_dict)}")
    
    # 2. 检查 aggregator 相关权重的统计
    print("\\n[2] Aggregator 权重统计:")
    agg_keys = [k for k in state_dict.keys() if k.startswith('aggregator.')]
    
    for key in agg_keys[:15]:  # 只显示前15个
        tensor = state_dict[key]
        print(f"    {key}:")
        print(f"      shape: {tensor.shape}")
        print(f"      mean: {tensor.float().mean().item():.6f}")
        print(f"      std: {tensor.float().std().item():.6f}")
        print(f"      min: {tensor.float().min().item():.6f}")
        print(f"      max: {tensor.float().max().item():.6f}")
    
    # 3. 特别关注 queries 和 proj_c
    print("\\n[3] 关键参数检查:")
    
    # queries
    queries_key = 'aggregator.boqs.0.queries'
    if queries_key in state_dict:
        q = state_dict[queries_key]
        print(f"    {queries_key}:")
        print(f"      shape: {q.shape}")
        print(f"      mean: {q.mean().item():.6f}")
        print(f"      std: {q.std().item():.6f}")
        print(f"      sample values: {q[0, 0, :5].tolist()}")
    
    # proj_c
    proj_key = 'aggregator.proj_c.weight'
    if proj_key in state_dict:
        p = state_dict[proj_key]
        print(f"    {proj_key}:")
        print(f"      shape: {p.shape}")
        print(f"      mean: {p.mean().item():.6f}")
        print(f"      std: {p.std().item():.6f}")
    
    # fc
    fc_key = 'aggregator.fc.weight'
    if fc_key in state_dict:
        f = state_dict[fc_key]
        print(f"    {fc_key}:")
        print(f"      shape: {f.shape}")
        print(f"      mean: {f.mean().item():.6f}")
        print(f"      std: {f.std().item():.6f}")
    
    # 4. 检查 backbone 权重
    print("\\n[4] Backbone 权重抽样检查:")
    backbone_keys = [k for k in state_dict.keys() if k.startswith('backbone.')]
    sample_key = backbone_keys[0] if backbone_keys else None
    if sample_key:
        b = state_dict[sample_key]
        print(f"    {sample_key}:")
        print(f"      shape: {b.shape}")
        print(f"      mean: {b.float().mean().item():.6f}")
        print(f"      std: {b.float().std().item():.6f}")
    
    # 5. 判断权重是否正常
    print("\\n[5] 诊断结论:")
    
    # 检查是否有明显的零值或极小值
    all_zeros = True
    for key in agg_keys[:5]:
        t = state_dict[key].float()
        if t.std().item() > 0.001:
            all_zeros = False
            break
    
    if all_zeros:
        print("    ❌ 警告：前5个Aggregator参数std都<0.001，权重可能已损坏")
    else:
        print("    ✅ 权重文件参数分布正常")
    
    # 检查文件大小
    import os
    file_size = os.path.getsize(checkpoint_path)
    print(f"    文件大小: {file_size / 1024 / 1024:.1f} MB")
    if file_size < 100 * 1024 * 1024:  # < 100MB
        print("    ❌ 警告：文件过小，可能下载不完整")
    else:
        print("    ✅ 文件大小正常")
    
    print("="*70)

if __name__ == "__main__":
    deep_verify()
'''

with open('/mnt/agents/output/deep_verify.py', 'w') as f:
    f.write(script_content)

print("诊断脚本已保存到: /mnt/agents/output/deep_verify.py")
print("\n请在终端运行: python deep_verify.py")

