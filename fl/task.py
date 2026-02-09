"""FL: A Flower / PyTorch app."""

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor, Resize

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch.optim as optim
from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.utils.ops import xywh2xyxy
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from flwr.common import Array, ArrayRecord
import torch
from torchvision.ops import box_iou, box_convert
from datetime import datetime

from flwr.common.logger import log, configure

configure(
    identifier="task",
    filename="logs/test_custom.log"
    # filename="logs/ODDnet_fedYogi_alpha=0.5_voc_data_Yogilr_e-4.log"
    # filename = f"logs/task_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

def get_model_record(model):
    """
    统一的转换函数：将模型转为 ArrayRecord，并强制修复标量形状。
    用于：
    1. ServerApp 初始化 Strategy
    2. ClientApp 返回参数
    """
    state_dict = model.state_dict()
    input_data_for_record = {}

    for name, param in state_dict.items():
        # 转为 numpy
        arr = param.detach().cpu().numpy()
        
        # [核心修复] 强制升维标量
        if arr.ndim == 0:
            arr = arr.reshape(1)
            
        # 封装为 Array
        input_data_for_record[name] = Array(arr)

    return ArrayRecord(input_data_for_record)

import torch

def load_rtdetr_weights(target_model, weight_path):
    """
    通用加载函数：支持加载官方权重或微调过的权重。
    会自动忽略形状不匹配的层（例如类别数不同的 Head 层）。
    """
    print(f"📂 正在加载权重文件: {weight_path}")
    
    # 1. 加载文件
    # map_location='cpu' 防止显存爆炸，之后再 .to(device)
    ckpt = torch.load(weight_path, map_location='cpu', weights_only=False)
    
    # 2. 智能提取权重 (优先使用 EMA，因为推理效果更好)
    if 'ema' in ckpt and ckpt['ema'] is not None:
        print("✨ 发现 EMA 权重，正在提取 (最佳推理性能)...")
        source_model = ckpt['ema']
    elif 'model' in ckpt:
        print("📦 提取常规 Model 权重...")
        source_model = ckpt['model']
    else:
        source_model = ckpt # 假设文件本身就是 state_dict
        
    # 获取 source state_dict
    # 如果 source_model 是一个完整模型对象(nn.Module)，取其 .state_dict()
    if hasattr(source_model, 'state_dict'):
        source_sd = source_model.state_dict()
    else:
        source_sd = source_model

    # 3. 准备目标 state_dict
    target_sd = target_model.state_dict()
    
    # 4. 核心逻辑：过滤匹配的权重
    # 只有当 Key 存在且 Shape 完全一致时才加载
    filtered_sd = {}
    mismatched_keys = []
    
    for k, v in source_sd.items():
        if k in target_sd:
            if v.shape == target_sd[k].shape:
                filtered_sd[k] = v
            else:
                mismatched_keys.append(f"{k} (源: {v.shape} vs 目标: {target_sd[k].shape})")
    
    # 5. 加载权重
    # strict=False 允许跳过不匹配的层
    target_model.load_state_dict(filtered_sd, strict=False)
    
    # 6. 打印报告
    print("=" * 40)
    print(f"🎉 成功加载参数: {len(filtered_sd)} / {len(target_sd)}")
    if len(mismatched_keys) > 0:
        print(f"⚠️ 跳过 {len(mismatched_keys)} 层 (通常是因为 nc 类别数不同):")
        # 只打印前3个跳过的层，避免刷屏
        for k in mismatched_keys[:3]:
            print(f"   - {k}")
        if len(mismatched_keys) > 3:
            print(f"   - ... 以及其他 {len(mismatched_keys)-3} 层")
    else:
        print("✅ 完美匹配，所有层均已加载！")
    print("=" * 40)
    
    return target_model    

def rtdetr_collate_fn(batch):
    """
    Final Robust Collate Function for RT-DETR
    """
    pixel_values = []
    batch_bboxes = []
    batch_cls = []
    gt_groups = []
    batch_idx_list = []
    
    for i, item in enumerate(batch):
        # 1. 图片堆叠
        pixel_values.append(item['pixel_values'])
        
        # 2. 标签处理
        # item['labels'] 应该是 apply_transforms 返回的 list
        # 格式: [[class, x, y, w, h], ...]
        labels = item['labels']
        num_gt = len(labels)
        gt_groups.append(num_gt)
        
        if num_gt > 0:
            labels_tensor = torch.tensor(labels, dtype=torch.float32)
            
            # 这里的转换非常关键
            cls = labels_tensor[:, 0].long()   # 必须是 Long
            bboxes = labels_tensor[:, 1:]      # 必须是 Float32
            
            # 检查坐标归一化 (防呆设计)
            if bboxes.max() > 1.0 + 1e-6:
                print(f"⚠️ Warning in Collate: Found bbox > 1.0 in image {i}, normalizing...")
                bboxes = torch.clamp(bboxes, 0.0, 1.0)
                
            batch_cls.append(cls)
            batch_bboxes.append(bboxes)
            
            # ⚠️ 关键修复：batch_idx 必须是 Long (int64) 类型
            # 形状要和 num_gt 一致
            b_idx = torch.full((num_gt,), i, dtype=torch.long)
            batch_idx_list.append(b_idx)
            
    # Stack images
    pixel_values = torch.stack(pixel_values, dim=0)
    
    # Flatten labels
    if len(batch_cls) > 0:
        targets_cls = torch.cat(batch_cls, dim=0)
        targets_bboxes = torch.cat(batch_bboxes, dim=0)
        targets_batch_idx = torch.cat(batch_idx_list, dim=0)
    else:
        # 空 Batch 处理
        targets_cls = torch.zeros(0, dtype=torch.long)
        targets_bboxes = torch.zeros(0, 4, dtype=torch.float32)
        targets_batch_idx = torch.zeros(0, dtype=torch.long)

    # 完整性校验 (Catch the error BEFORE model forward)
    assert len(targets_cls) == len(targets_bboxes) == len(targets_batch_idx) == sum(gt_groups), \
        "Data Mismatch! The collate function failed to align targets."

    return {
        'images': pixel_values,
        'cls': targets_cls,
        'bboxes': targets_bboxes,
        'gt_groups': gt_groups,
        'batch_idx': targets_batch_idx
    }

fds = None  # Cache FederatedDataset
# Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
pytorch_transforms = A.Compose([
    A.LongestMaxSize(max_size=640),
    A.PadIfNeeded(min_height=640, min_width=640, border_mode=0, value=(114, 114, 114)),
    A.Normalize(mean=(0,0,0), std=(1,1,1)), # 相当于 / 255.0
    ToTensorV2()
], bbox_params=A.BboxParams(format='yolo', min_visibility=0.1))

# def apply_transforms(batch):
#     """Apply transforms to the partition from FederatedDataset."""
#     batch["pixel_values"] = [pytorch_transforms(img.convert("RGB")) for img in batch["image"]]
#     return batch

def apply_transforms(batch):
    """
    适配 Albumentations 的 transform 函数。
    输入 batch['labels'] 格式: List[List[class_id, x, y, w, h]]
    """
    pixel_values = []
    new_labels = []

    for img, labels in zip(batch["image"], batch["labels"]):
        # 1. 图片转 Numpy (Albumentations 不接受 PIL)
        image_np = np.array(img.convert("RGB"))

        # 2. 标签格式转换
        # 你的数据: [class, x, y, w, h]
        # Albumentations 需要将 class_id 分离，或者放在最后 [x, y, w, h, class_id]
        # 我们这里采用把 BBox 和 Class 分离传入的方式，最稳妥
        bboxes_only = []
        class_labels = []
        
        for item in labels:
            c, x, y, w, h = item
            # 必须确保坐标在 0-1 之间，防止数值误差导致的报错
            # 虽然你检查过了，加个保险不亏
            x = min(max(x, 0.0), 1.0)
            y = min(max(y, 0.0), 1.0)
            w = min(max(w, 0.0), 1.0)
            h = min(max(h, 0.0), 1.0)
            
            bboxes_only.append([x, y, w, h, c]) # 临时把 c 放在最后传给 transform

        # 3. 执行变换
        # 注意: 必须使用关键字参数 image=, bboxes=
        try:
            transformed = pytorch_transforms(image=image_np, bboxes=bboxes_only)
        except ValueError as e:
            # 如果某个 bbox 数据严重错误导致变换失败，捕获异常并跳过该图（或打印警告）
            print(f"⚠️ Transform failed for an image: {e}")
            # 返回全黑图和空标签防止程序崩溃
            pixel_values.append(torch.zeros(3, 640, 640))
            new_labels.append([])
            continue

        # 4. 提取结果
        transformed_image = transformed["image"] # 已经是 Tensor
        transformed_bboxes = transformed["bboxes"] # 变换后的 bbox

        # 5. 将标签还原回 [class, x, y, w, h] 格式供 collate_fn 使用
        final_labels = []
        for bbox in transformed_bboxes:
            # Albumentations 返回的格式与传入一致: [x, y, w, h, c]
            x, y, w, h, c = bbox
            final_labels.append([c, x, y, w, h])

        pixel_values.append(transformed_image)
        new_labels.append(final_labels)

    # 更新 batch
    batch["pixel_values"] = pixel_values
    batch["labels"] = new_labels
    return batch

# def load_data(partition_id: int, num_partitions: int):
#     """Load partition data."""
#     # Only initialize `FederatedDataset` once
#     global fds
#     if fds is None:
#         partitioner = IidPartitioner(num_partitions=num_partitions)
#         # partitioner = DirichletPartitioner(num_partitions=num_partitions, partition_by="partition_cls",
#         #                            alpha=0.5, min_partition_size=10,
#         #                            self_balancing=True)
#         fds_train = FederatedDataset(
#             dataset="/root/fl/datasets/coco8/coco8_load_train.py",
#             partitioners={"train": partitioner},
#             data_dir="/root/fl/datasets/VOC2007",
#             # data_dir="/root/autodl-tmp/local",
#             trust_remote_code=True,
#             # download_mode="force_redownload"
#         )

#         fds_test = FederatedDataset(
#             dataset="/root/fl/datasets/coco8/coco8_load_test.py",
#             partitioners={"test": partitioner},
#             data_dir="/root/fl/datasets/VOC2007",
#             trust_remote_code=True,
#         )

#     partition_train = fds_train.load_partition(partition_id)
#     partition_test = fds_test.load_partition(partition_id)
#     # Divide data on each node: 80% train, 20% test
#     partition_train = partition_train.train_test_split(test_size=0.01, seed=42)
#     partition_test = partition_test.train_test_split(test_size=0.99, seed=42)
#     # Construct dataloaders
#     partition_train = partition_train.with_transform(apply_transforms)
#     partition_test = partition_test.with_transform(apply_transforms)
#     trainloader = DataLoader(
#         partition_train["train"], 
#         batch_size=4, 
#         shuffle=True,
#         collate_fn=rtdetr_collate_fn
#     )
#     testloader = DataLoader(
#         partition_test["test"], 
#         batch_size=4,
#         shuffle=False,
#         collate_fn=rtdetr_collate_fn
#     )
#     return trainloader, testloader

# 缓存对象
fds_train, fds_test = None, None

def load_data(partition_id: int, num_partitions: int):
    global fds_train, fds_test
    if fds_train is None or fds_test is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds_train = FederatedDataset(
            dataset="/root/fl/datasets/coco8/coco8_load_train.py",
            partitioners={"train": partitioner},
            data_dir="/root/autodl-tmp/local",
            trust_remote_code=True,
        )
        fds_test = FederatedDataset(
            dataset="/root/fl/datasets/coco8/coco8_load_test.py",
            partitioners={"test": partitioner},
            data_dir="/root/autodl-tmp/local",
            trust_remote_code=True,
        )

    # 显式指定 split，避免默认走第一个分区器
    partition_train = fds_train.load_partition(partition_id, split="train")
    partition_test = fds_test.load_partition(partition_id, split="test")

    partition_train = partition_train.with_transform(apply_transforms)
    partition_test = partition_test.with_transform(apply_transforms)

    trainloader = DataLoader(partition_train, batch_size=4, shuffle=True, collate_fn=rtdetr_collate_fn)
    testloader = DataLoader(partition_test, batch_size=4, shuffle=False, collate_fn=rtdetr_collate_fn)
    return trainloader, testloader

from .myutils import visualize_batch

def train(net, trainloader, epochs, lr, device):
    """
    Train the model on the training set using RT-DETR loss.
    Returns: avg_trainloss (float)
    """
    net.train()
    
    # 同时使用VFL和EQLv2
    criterion = RTDETRDetectionLoss(
        nc=20, 
        use_vfl=True,       # 保持VFL
        use_eqlv2=True,     # 同时启用EQLv2
        loss_gain = {
            "class": 0.5,
            "eqlv2": 0.5,      
            "bbox": 5.0,
            "giou": 2.0,
            "no_object": 0.1,
            "mask": 1.0,
            "dice": 1.0,
        },
        gamma=1.5,
        alpha=0.25, 
        eql_gamma=12.0,
        eql_mu=0.8,
        eql_alpha=4.0,
    ).to(device)
    
    # 最佳实践：RT-DETR 推荐使用 AdamW 和 weight decay
    # optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=0.0001)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, net.parameters()), 
        lr=0.0001
    )
    
    total_loss = 0.0
    num_batches = 0

    vis_debug_max = 3  # 只在首个 epoch 保存前3个批次
    vis_debug_done = 0
    
    for epoch in range(epochs):
        for batch in trainloader:

            if epoch == 0 and vis_debug_done < vis_debug_max:
                visualize_batch(
                    batch,
                    save_dir="logs/vis",
                    prefix=f"epoch{epoch}_batch{vis_debug_done}",
                    class_names=None,  # 若有类别名列表可填上
                    max_images=None
                )
                vis_debug_done += 1

            # 1. 数据迁移
            images = batch['images'].to(device)
            batch['cls'] = batch['cls'].to(device)
            batch['bboxes'] = batch['bboxes'].to(device)
            batch['batch_idx'] = batch['batch_idx'].to(device) # <--- 关键：CDN需要
            
            optimizer.zero_grad()
            
            # 2. Forward (传入 batch 用于生成去噪锚框)
            outputs = net(images, batch=batch)
            
            # 3. Loss 计算
            # 解包: x = (dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta)
            dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = outputs

            if enc_bboxes is not None:
                enc_bboxes = enc_bboxes.unsqueeze(0)
                enc_scores = enc_scores.unsqueeze(0)
            
            loss_dict = criterion(
                preds=(dec_bboxes, dec_scores),
                batch=batch,
                dn_bboxes=enc_bboxes,
                dn_scores=enc_scores,
                dn_meta=dn_meta
            )
            
            # print(f"losses: {loss_dict.values()}")

            # 4. Backward
            loss = sum(loss_dict.values())
            loss.backward()
            
            # 最佳实践：梯度裁剪防止爆炸
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.1)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
    avg_trainloss = total_loss / max(num_batches, 1)
    return avg_trainloss


# def test(net, testloader, device):
#     """
#     Validate the model.
#     Returns: loss (float), map50-95 (float)
#     """
#     net.to(device)
#     net.eval()
    
#     criterion = RTDETRDetectionLoss(nc=20, use_vfl=True).to(device)
#     metric = MeanAveragePrecision(box_format="cxcywh", iou_type="bbox").to(device)
    
#     total_loss = 0.0
#     num_batches = 0
    
#     with torch.no_grad():
#         for batch in testloader:
#             images = batch['images'].to(device)
#             batch['cls'] = batch['cls'].to(device)
#             batch['bboxes'] = batch['bboxes'].to(device)
#             # Eval 模式不需要 batch_idx 用于 CDN，但如果代码没改干净，移过去也无妨
            
#             # 1. Forward
#             # Eval模式下，你的forward返回 (y, x)
#             # y: (BS, 300, 4+nc) 推理结果
#             # x: (dec_bboxes, ...) 原始输出用于算 Loss
#             outputs = net(images, batch=batch)
            
#             # 确保我们拿到了正确的输出
#             if isinstance(outputs, tuple) and len(outputs) == 2:
#                 inference_out, raw_out = outputs
#                 dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = raw_out

#                 if enc_bboxes is not None:
#                     enc_bboxes = enc_bboxes.unsqueeze(0)
#                     enc_scores = enc_scores.unsqueeze(0)
                    
#             else:
#                 # 兼容性 fallback，防止 return 结构变化
#                 continue

#             # 2. 计算 Validation Loss
#             loss_dict = criterion(
#                 preds=(dec_bboxes, dec_scores),
#                 batch=batch,
#                 dn_bboxes=enc_bboxes,
#                 dn_scores=enc_scores,
#                 dn_meta=dn_meta
#             )
#             total_loss += sum(loss_dict.values()).item()
#             num_batches += 1
            
#             # 3. 计算 mAP (Best Practice)
#             # 需要将 flattened 的 targets 还原回 per-image 格式
#             target_list = []
#             current_idx = 0
#             for num_gt in batch['gt_groups']:
#                 if num_gt > 0:
#                     t_boxes = batch['bboxes'][current_idx : current_idx + num_gt]
#                     t_labels = batch['cls'][current_idx : current_idx + num_gt]
#                     target_list.append(dict(boxes=t_boxes, labels=t_labels))
#                     current_idx += num_gt
#                 else:
#                     target_list.append(dict(boxes=torch.empty(0, 4, device=device), labels=torch.empty(0, device=device)))

#             # 解析预测结果 inference_out: (BS, 300, 4+nc)
#             # 格式通常是 [cx, cy, w, h, class_probs...] 或者 [cx, cy, w, h, max_score, class_id]
#             # 根据你 forward 的最后一行： torch.cat((bboxes, scores), -1)
#             # 前4位是 bbox, 后面是 scores
#             pred_list = []
#             bs = inference_out.shape[0]
#             for i in range(bs):
#                 pred_item = inference_out[i]
#                 p_boxes = pred_item[:, :4] # cx, cy, w, h
#                 p_scores = pred_item[:, 4:] # 80个类别的分数
                
#                 # 获取每个 box 的最大分数和对应类别
#                 scores, labels = p_scores.max(dim=-1)
                
#                 pred_list.append(dict(
#                     boxes=p_boxes,
#                     scores=scores,
#                     labels=labels
#                 ))
            
#             # 更新指标状态
#             metric.update(pred_list, target_list)

#     # 计算最终 mAP
#     metrics_dict = metric.compute()
#     map50 = metrics_dict['map_50'].item() # map 默认就是 mAP 50-95
#     avg_loss = total_loss / max(num_batches, 1)
    
#     # 这里的 map50_95 替代了原本的 accuracy
#     return avg_loss, map50

def test(net, testloader, device):
    """
        Validate the model.
        自动寻找最佳 F1 阈值。  
        Returns: 
            loss (float), map50 (float), 
            best_precision (float), best_recall (float), 
            best_f1 (float), best_threshold (float)
    """
    net.to(device)
    net.eval()
    
    criterion = RTDETRDetectionLoss(nc=20, use_vfl=True).to(device)
    metric = MeanAveragePrecision(box_format="cxcywh", iou_type="bbox").to(device)
    
    total_loss = 0.0
    num_batches = 0
    
    # 存储所有预测结果用于后续计算最佳阈值
    # 格式: list of tensors [score, is_tp]
    # is_tp: 1 if TP, 0 if FP
    pred_stats = [] 
    total_gt_count = 0 # 整个数据集的 GT 总数
    
    with torch.no_grad():
        for batch in testloader:
            images = batch['images'].to(device)
            batch['cls'] = batch['cls'].to(device)
            batch['bboxes'] = batch['bboxes'].to(device)
            
            # 1. Forward
            outputs = net(images, batch=batch)
            
            if isinstance(outputs, tuple) and len(outputs) == 2:
                inference_out, raw_out = outputs
                dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = raw_out
                if enc_bboxes is not None:
                    enc_bboxes = enc_bboxes.unsqueeze(0)
                    enc_scores = enc_scores.unsqueeze(0)
            else:
                continue

            # 2. Loss
            loss_dict = criterion(
                preds=(dec_bboxes, dec_scores),
                batch=batch,
                dn_bboxes=enc_bboxes,
                dn_scores=enc_scores,
                dn_meta=dn_meta
            )
            total_loss += sum(loss_dict.values()).item()
            num_batches += 1
            
            # 3. 准备数据
            target_list = []
            current_idx = 0
            for num_gt in batch['gt_groups']:
                if num_gt > 0:
                    t_boxes = batch['bboxes'][current_idx : current_idx + num_gt]
                    t_labels = batch['cls'][current_idx : current_idx + num_gt]
                    target_list.append(dict(boxes=t_boxes, labels=t_labels))
                    current_idx += num_gt
                    total_gt_count += num_gt # 累计 GT 总数
                else:
                    target_list.append(dict(boxes=torch.empty(0, 4, device=device), labels=torch.empty(0, device=device)))

            pred_list = []
            bs = inference_out.shape[0]
            
            # --- 处理每一张图片，收集 TP/FP 状态 ---
            for i in range(bs):
                # A. 解析预测
                pred_item = inference_out[i]
                p_boxes = pred_item[:, :4] 
                p_scores_all = pred_item[:, 4:] 
                scores, labels = p_scores_all.max(dim=-1)
                
                # B. 加入 metric 更新队列 (计算 mAP 用)
                pred_list.append(dict(boxes=p_boxes, scores=scores, labels=labels))
                
                # C. 手动匹配逻辑 (计算 Best F1 用)
                # 即使分数很低也要保留，因为我们要画完整曲线，但为了显存可以设个极低门槛
                keep_mask = scores > 0.001 
                filter_boxes = p_boxes[keep_mask]
                filter_scores = scores[keep_mask]
                
                gt_boxes = target_list[i]['boxes']
                
                # 如果没有预测框
                if len(filter_boxes) == 0:
                    continue
                
                # 记录当前图片的预测状态: [score, 0/1]
                # 默认为 FP (0)
                matches = torch.zeros(len(filter_boxes), device=device) 
                
                if len(gt_boxes) > 0:
                    # 转换坐标 cxcywh -> xyxy
                    p_xyxy = box_convert(filter_boxes, in_fmt='cxcywh', out_fmt='xyxy')
                    g_xyxy = box_convert(gt_boxes, in_fmt='cxcywh', out_fmt='xyxy')
                    
                    # 计算 IoU: [N_pred, M_gt]
                    iou_matrix = box_iou(p_xyxy, g_xyxy)
                    
                    # 按分数从高到低排序预测框，进行贪婪匹配
                    sort_idx = torch.argsort(filter_scores, descending=True)
                    iou_matrix = iou_matrix[sort_idx]
                    
                    gt_matched = torch.zeros(len(gt_boxes), dtype=torch.bool, device=device)
                    
                    # 这是一个临时的 vector 对应排序后的 indices
                    sorted_matches = torch.zeros(len(filter_boxes), device=device)
                    
                    for p_idx in range(len(filter_boxes)):
                        # 找到该预测框最大 IoU 的 GT
                        iou_val, g_idx = iou_matrix[p_idx].max(0)
                        
                        if iou_val > 0.5 and not gt_matched[g_idx]:
                            gt_matched[g_idx] = True
                            sorted_matches[p_idx] = 1.0 # 标记为 TP
                    
                    # 恢复原来的顺序 (或者直接存排序后的 score 也行，这里为了简单直接存排序后的)
                    # 我们只需要 score 和是否 TP 的对应关系
                    pred_stats.append(torch.stack((filter_scores[sort_idx], sorted_matches), dim=1))
                
                else:
                    # 图片没有 GT，所有预测框都是 FP (matches 全 0)
                    pred_stats.append(torch.stack((filter_scores, matches), dim=1))

            metric.update(pred_list, target_list)

    # --- Loop 结束，开始计算指标 ---
    
    # 1. 计算标准 mAP
    metrics_dict = metric.compute()
    map50 = metrics_dict['map_50'].item()
    avg_loss = total_loss / max(num_batches, 1)

    # 2. 计算最佳 F1 及其对应的 P, R, Threshold
    if len(pred_stats) > 0:
        # 拼接所有 batch 的数据: shape [N_total_preds, 2]
        all_stats = torch.cat(pred_stats, dim=0)
        
        # 按分数从高到低排序
        sorted_indices = torch.argsort(all_stats[:, 0], descending=True)
        sorted_stats = all_stats[sorted_indices]
        
        all_scores = sorted_stats[:, 0]
        all_tps = sorted_stats[:, 1] # 1.0 or 0.0
        
        # 向量化计算累积 TP 和 FP
        # cumsum 告诉我们在当前 index (即当前阈值) 下，有多少个 TP 和 FP
        tp_cumsum = torch.cumsum(all_tps, dim=0)
        fp_cumsum = torch.cumsum(1 - all_tps, dim=0)
        
        # 防止除零
        eps = 1e-7
        
        # 计算 Precision 和 Recall 曲线
        precision_curve = tp_cumsum / (tp_cumsum + fp_cumsum + eps)
        recall_curve = tp_cumsum / (total_gt_count + eps)
        
        # 计算 F1 曲线
        f1_curve = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve + eps)
        
        # 找到 F1 最大的位置
        best_idx = torch.argmax(f1_curve)
        
        best_f1 = f1_curve[best_idx].item()
        best_precision = precision_curve[best_idx].item()
        best_recall = recall_curve[best_idx].item()
        best_threshold = all_scores[best_idx].item()
        
    else:
        # 防止验证集为空或没有预测框的极端情况
        best_f1, best_precision, best_recall, best_threshold = 0.0, 0.0, 0.0, 0.0

    print(f"Best F1: {best_f1:.4f} @ Threshold: {best_threshold:.4f} (P={best_precision:.4f}, R={best_recall:.4f})")
    
    return avg_loss, map50, best_precision, best_recall, best_f1, best_threshold


# def train(net, trainloader, epochs, lr, device):
#     """Train the model on the training set."""
#     net.to(device)  # move model to GPU if available
#     criterion = torch.nn.CrossEntropyLoss().to(device)
#     optimizer = torch.optim.Adam(net.parameters(), lr=lr)
#     net.train()
#     running_loss = 0.0
#     for _ in range(epochs):
#         for batch in trainloader:
#             images = batch["images"].to(device)
#             labels = batch["labels"].to(device)
#             optimizer.zero_grad()
#             loss = criterion(net(images), labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#     avg_trainloss = running_loss / len(trainloader)
#     return avg_trainloss


# def test(net, testloader, device):
#     """Validate the model on the test set."""
#     net.to(device)
#     net.eval()
#     criterion = torch.nn.CrossEntropyLoss()
#     correct, loss = 0, 0.0
#     with torch.no_grad():
#         for batch in testloader:
#             images = batch["images"].to(device)
#             labels = batch["labels"].to(device)
#             outputs = net(images)
#             loss += criterion(outputs, labels).item()
#             correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
#     accuracy = correct / len(testloader.dataset)
#     loss = loss / len(testloader)
#     return loss, accuracy
