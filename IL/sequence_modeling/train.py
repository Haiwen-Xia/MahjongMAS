import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter # type:ignore[import]
from dataset import get_mahjong_dataloader
from model import TimeNet
import argparse
from tqdm import tqdm
import ipdb
def parse_args():
    parser = argparse.ArgumentParser(description="麻将AI模型训练")
    parser.add_argument("--data_dir", type=str, default='data', help="数据目录")
    parser.add_argument("--log_dir", type=str, default="logs", help="日志目录")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="模型保存目录")
    parser.add_argument("--batch_size", type=int, default=32, help="批处理大小")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--hid_dim", type=int, default=512, help="隐藏层维度")
    parser.add_argument("--n_heads", type=int, default=8, help="注意力头数")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout率")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="设备")
    parser.add_argument("--num_layers", type=int, default=6, help="Transformer层数")
    parser.add_argument("--dataset", type=str, default="batch_file", choices=["efficient", "time","batch_file"], help="数据集类型")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="预热步数占总步数的比例")
    parser.add_argument("--scheduler", type=str, default="cosine_warmup", 
                        choices=["cosine_warmup", "linear_warmup", "constant"], 
                        help="学习率调度策略")    
    return parser.parse_args()

def train_epoch(model, dataloader, optimizer, device, epoch, writer, scheduler=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    for i, batch in enumerate(pbar):
        # 将数据移至设备
        history = batch["history"].to(device)
        global_state = batch["global_state"].to(device)
        action_mask = batch["action_mask"].to(device)
        action = batch["action"].to(device)
        pad_mask = batch["pad_mask"].to(device)
        # for key, value in batch.items():
        #     if isinstance(value, torch.Tensor):
        #         if torch.isnan(value).any():
        #             print(f"Error: NaN found in batch['{key}'] at step {i}")
        #             ipdb.set_trace()
        #         if torch.isinf(value).any():
        #             print(f"Error: Inf found in batch['{key}'] at step {i}")
        #             ipdb.set_trace()
        # 构建模型输入
        model_input = {
            "history": history,
            "global_state": global_state,
            "action_mask": action_mask
        }
        
        # 前向传播
        optimizer.zero_grad()
        logits = model(model_input, pad_mask)
        
        # 计算损失
        loss = F.cross_entropy(logits, action)
        
        # 反向传播
        loss.backward()
        # grad_clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)        
        optimizer.step()
        # 更新学习率
        if scheduler is not None:
            scheduler.step()
            # 记录当前学习率
            current_lr = scheduler.get_last_lr()[0]
            writer.add_scalar("train/learning_rate", current_lr, epoch * len(dataloader) + i)
                
        # 统计
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == action).sum().item()
        total += action.size(0)
        
        # 更新进度条
        pbar.set_postfix({
            "loss": total_loss / (i + 1),
            "acc": correct / total,
            'norm': grad_norm,
        })
        
        # 记录到TensorBoard
        step = epoch * len(dataloader) + i
        writer.add_scalar("train/loss", loss.item(), step)
        writer.add_scalar("train/accuracy", correct / total, step)
        writer.add_scalar("train/grad_norm", grad_norm, step)

    return total_loss / len(dataloader), correct / total

def validate(model, dataloader, device, epoch, writer):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Validating")):
            # 将数据移至设备
            history = batch["history"].to(device)
            global_state = batch["global_state"].to(device)
            action_mask = batch["action_mask"].to(device)
            action = batch["action"].to(device)
            pad_mask = batch["pad_mask"].to(device)
            
            # 构建模型输入
            model_input = {
                "history": history,
                "global_state": global_state,
                "action_mask": action_mask
            }
            
            # 前向传播
            logits = model(model_input, pad_mask)
            
            # 计算损失
            loss = F.cross_entropy(logits, action)
            
            # 统计
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == action).sum().item()
            total += action.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    # 记录到TensorBoard
    writer.add_scalar("val/loss", avg_loss, epoch)
    writer.add_scalar("val/accuracy", accuracy, epoch)
    
    return avg_loss, accuracy

def main():
    args = parse_args()
    
    # 创建目录
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 初始化TensorBoard

    # Create a timestamped directory for logs
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_dir, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # 获取数据加载器
    train_loader = get_mahjong_dataloader(
        args.data_dir, 
        batch_size=args.batch_size, 
        split='train', 
        shuffle=True,
        name=args.dataset,
        num_workers=4
    )
    
    val_loader = get_mahjong_dataloader(
        args.data_dir, 
        batch_size=args.batch_size, 
        split='val', 
        shuffle=False,
        name=args.dataset
    )

    # 初始化模型
    # 将args中的模型相关参数转换为字典形式
    model_args = vars(args)
    print("以下为模型参数：\n",model_args)
    # 也可以只选择模型需要的特定参数
    # model_args = {k: v for k, v in vars(args).items() if k in ["hid_dim", "n_heads", "dropout", "device", "num_layers"]}
    import numpy as np
    example = np.load(os.path.join(args.data_dir, "0.npz"))
    history_s = example["history"].shape[-1]
    state_s = example["global_state"][-1].shape
    print(history_s, state_s)
    model = TimeNet(
        history_feature_dim=history_s, 
        state_feature_shape=state_s, 
        output_dim=235,
        args=model_args
    ).to(args.device)
    
    # 初始化优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    if args.scheduler == "cosine_warmup":
        from transformers import get_cosine_schedule_with_warmup # type:ignore[import]
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=total_steps
        )
    elif args.scheduler == "linear_warmup":
        from transformers import get_linear_schedule_with_warmup # type:ignore[import]
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
    else:  # constant
        scheduler = None
            
    # 训练循环
    best_val_acc = 0
    for epoch in range(args.epochs):
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, args.device, epoch, writer, scheduler
        )
        
        # 验证
        val_loss, val_acc = validate(
            model, val_loader, args.device, epoch, writer
        )
        
        print(f"Epoch {epoch+1}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, os.path.join(args.save_dir, 'best_model.pt'))
        
        # 定期保存检查点
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, os.path.join(args.save_dir, f'checkpoint_ep{epoch+1}.pt'))
    
    writer.close()
    print(f"训练完成! 最佳验证准确率: {best_val_acc:.4f}")

if __name__ == "__main__":
    main()