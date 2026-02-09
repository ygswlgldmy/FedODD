import torch
import sys
import os

# ================= 配置区域 =================
# 1. 这里填你的 .pt 文件路径
PT_PATH = "/root/fl/ultralytics/runs/detect/rtdetrl_voc_finetuned3/weights/best.pt"

# 2. 导入你自定义的模型类 (请根据你的实际项目结构修改导入路径)
# 假设你的类在 model.py 中，类名叫 RTDETR_L
# from model import RTDETR_L  <-- 请取消注释并修改这一行
# ⚠️ 为了演示，如果你没有导入，我在下面用一个伪代码占位，实际运行时请确保你能实例化模型
try:
    from models import RTDETR_L # 举例：请修改为你实际的 import 路径
    print("✅ 成功导入 RTDETR_L 类")
except ImportError:
    print("❌ 警告：无法导入 RTDETR_L 类。请修改脚本中的 import 语句。")
    print("   暂时无法对比目标模型结构，仅分析 .pt 文件。")
    RTDETR_L = None
# ===========================================

def analyze_checkpoint():
    print(f"\n{'='*20} 开始分析 .pt 文件 {'='*20}")
    print(f"📂 文件路径: {PT_PATH}")

    try:
        # 1. 加载 Checkpoint
        # map_location='cpu' 保证在任何机器上都能跑
        # weights_only=False 是关键，用于解决 PyTorch 2.6+ 加载 Ultralytics 模型的报错
        ckpt = torch.load(PT_PATH, map_location='cpu', weights_only=False)
        print(f"✅ 文件加载成功！类型: {type(ckpt)}")
    except Exception as e:
        print(f"❌ 文件加载失败: {e}")
        return

    # 2. 分析顶层键
    if isinstance(ckpt, dict):
        print(f"🔑 顶层 Keys: {list(ckpt.keys())}")
        
        # 智能选择最佳权重源
        source_key = None
        if 'ema' in ckpt:
            print("✨ 发现 'ema' (指数移动平均) 权重，这是推理的最佳选择。")
            source_key = 'ema'
        elif 'model' in ckpt:
            print("ℹ️ 发现 'model' 权重。")
            source_key = 'model'
        else:
            print("⚠️ 未发现 standard keys，假设整个字典就是权重。")
            weights = ckpt

        if source_key:
            weights = ckpt[source_key]
    else:
        # 这种情况极少见，除非保存的直接是 state_dict
        print("⚠️ Checkpoint 本身不是字典，可能直接是模型对象或 state_dict。")
        weights = ckpt

    # 3. 提取 State Dict
    # Ultralytics 的 value 通常是完整的 nn.Module 对象，需要 .state_dict()
    if hasattr(weights, 'state_dict'):
        print(f"📦 权重源 ({source_key}) 是一个模型对象 (nn.Module)，正在提取 state_dict...")
        pt_state_dict = weights.state_dict()
    elif isinstance(weights, dict):
        print(f"📄 权重源 ({source_key}) 已经是一个字典。")
        pt_state_dict = weights
    else:
        print(f"❌ 无法识别权重格式: {type(weights)}")
        return

    pt_keys = list(pt_state_dict.keys())
    print(f"📊 .pt 文件中包含 {len(pt_keys)} 个参数层。")

    # 4. 如果用户定义了本地模型，进行对比
    if RTDETR_L:
        try:
            model = RTDETR_L()
            model_keys = list(model.state_dict().keys())
            print(f"\n{'='*20} 结构对比 {'='*20}")
            print(f"🏗️  你的自定义模型 (RTDETR_L) 包含 {len(model_keys)} 个参数层。")

            print("\n--- 🔍 前 5 层名称对比 ---")
            print(f"{'Source (.pt)':<50} | {'Target (Your Class)':<50}")
            print("-" * 105)
            for i in range(min(5, len(pt_keys), len(model_keys))):
                print(f"{pt_keys[i]:<50} | {model_keys[i]:<50}")
            
            # 5. 自动前缀检测
            sample_pt = pt_keys[0]
            sample_model = model_keys[0]
            
            # 简单启发式检查
            if sample_pt == sample_model:
                print("\n✅以此看来，层名称完全匹配！可以直接加载。")
            else:
                print("\n⚠️ 发现层名称不匹配！")
                common_prefixes = ['module.', 'model.', 'base.']
                found_prefix = False
                for prefix in common_prefixes:
                    if sample_pt.startswith(prefix) and sample_pt.replace(prefix, "") == sample_model:
                        print(f"💡 诊断: .pt 文件中的层多了一个前缀 '{prefix}'")
                        print(f"   解决: 在加载时剔除 '{prefix}' 即可。")
                        found_prefix = True
                        break
                    if sample_model.startswith(prefix) and sample_model.replace(prefix, "") == sample_pt:
                        print(f"💡 诊断: 你的模型定义多了一个前缀 '{prefix}'")
                        found_prefix = True
                        break
                
                if not found_prefix:
                     print("💡 诊断: 结构差异较大，可能需要正则匹配或手动重命名。")

        except Exception as e:
            print(f"❌ 实例化 RTDETR_L 失败: {e}")

if __name__ == "__main__":
    analyze_checkpoint()