import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter # type:ignore[import]
from dataset import get_mahjong_dataloader
from model import TimeNet
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="麻将AI模型训练")
    parser.add_argument("--data_dir", type=str, required=True, help="数据目录")
    parser.add_argument("--log_dir", type=str, default="logs", help="日志目录")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="模型保存目录")
    parser.add_argument("--batch_size", type=int, default=32, help="批处理大小")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--hid_dim", type=int, default=512, help="隐藏层维度")
    parser.add_argument("--n_heads", type=int, default=8, help="注意力头数")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout率")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="设备")
    return parser.parse_args()

def train_epoch(model, dataloader, optimizer, device, epoch, writer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    for i, batch in enumerate(pbar):
        # 将数据移至设备
        history = batch["history"].to(device)
        hand = batch["hand"].to(device)
        action_mask = batch["action_mask"].to(device)
        action = batch["action"].to(device)
        pad_mask = batch["pad_mask"].to(device)
        
        # 构建模型输入
        model_input = {
            "history": history,
            "hand": hand,
            "action_mask": action_mask
        }
        
        # 前向传播
        optimizer.zero_grad()
        logits = model(model_input, pad_mask)
        
        # 计算损失
        loss = F.cross_entropy(logits, action)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == action).sum().item()
        total += action.size(0)
        
        # 更新进度条
        pbar.set_postfix({
            "loss": total_loss / (i + 1),
            "acc": correct / total
        })
        
        # 记录到TensorBoard
        step = epoch * len(dataloader) + i
        writer.add_scalar("train/loss", loss.item(), step)
        writer.add_scalar("train/accuracy", correct / total, step)
    
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
            hand = batch["hand"].to(device)
            action_mask = batch["action_mask"].to(device)
            action = batch["action"].to(device)
            pad_mask = batch["pad_mask"].to(device)
            
            # 构建模型输入
            model_input = {
                "history": history,
                "hand": hand,
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
    writer = SummaryWriter(args.log_dir)
    
    # 获取数据加载器
    train_loader = get_mahjong_dataloader(
        args.data_dir, 
        batch_size=args.batch_size, 
        split='train', 
        shuffle=True
    )
    
    val_loader = get_mahjong_dataloader(
        args.data_dir, 
        batch_size=args.batch_size, 
        split='val', 
        shuffle=False
    )

    # 初始化模型
    model_args = {
        "hid_dim": args.hid_dim,
        "n_heads": args.n_heads,
        "dropout": args.dropout,
        "device": args.device
    }
    
    model = TimeNet(
        history_feature_dim=95, 
        state_feature_shape=(4, 4, 9), 
        output_dim=235,
        args=model_args
    ).to(args.device)
    
    # 初始化优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 训练循环
    best_val_acc = 0
    for epoch in range(args.epochs):
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, args.device, epoch, writer
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