import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from conv import ConvBlock, ResidualBlock
# ---------- 通用模块 ----------
class PositionalEncoding(nn.Module):
    """标准 Transformer sine‑cosine 位置编码（batch_first=True）"""
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)              # (L, D)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe 张量被注册为模型的一个缓冲区
        # 模型的缓冲区会随模型一起自动迁移到指定设备        
        self.register_buffer("pe", pe.unsqueeze(0))     # (1, L, D)

    def forward(self, x):
        return self.pe[:, : x.size(1)]

class AttentionBlock(nn.Module):
    """Residual‑Norm wrapper over MultiheadAttention (supports self & cross)."""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ln = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, q, k=None, v=None, key_padding_mask=None, attn_mask=None):
        k = q if k is None else k
        v = k if v is None else v
        # q/k/v shape: (B, T, D)
        out, _ = self.mha( # _ 为 attention weights
            q, k, v,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask    # 用于忽略 padding token
        )
        return self.ln(q + self.drop(out))       # Residual + LayerNorm
class TransformerBlock(nn.Module): # 4*512*512
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    def forward(self, x, key_padding_mask=None):
        attn_output, _ = self.attention(x, x, x, key_padding_mask=key_padding_mask)
        x = x + self.dropout1(attn_output)
        x = self.ln1(x)
        ff_output = self.ff(x)
        x = x + self.dropout2(ff_output)
        return self.ln2(x)
    
class Transformer(nn.Module):
    def __init__(self, d_model, n_heads, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, dropout ) for _ in range(num_layers)]
        )
    def forward(self, x, key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return x
# ---------- TimeNet ----------
class TimeNet(nn.Module):
    '''
    初步baseline架构：
    
    历史信息先embedding(Linear是最直接的，可能还可以尝试GLU等更非线性的)
    然后加上位置编码，接着用Transformer encoder处理
    
    当前信息仍用卷积提初级特征
    
    随后，把当前信息cat到历史信息的第一位，再作多次self attention,把它当作summary token
    
    最后接全连接
    
    mask由dataloader生成,pad 到同一个batch里面的最长者
    '''
    def __init__(
        self,
        history_feature_dim,        # 95 
        state_feature_shape,        # (C,H,W) = (4,4,9), C可以增加
        output_dim,
        args=None
    ):
        super().__init__()
        args = args or {}
        self.hid = args.get("hid_dim", 512)
        n_heads = args.get("n_heads", 8)
        drop = args.get("dropout", 0.1)
        device = args.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        # 1. embedding + position
        self.embed = nn.Sequential(
            nn.Linear(history_feature_dim, 4* self.hid),
            nn.ReLU(),
            nn.Linear(4* self.hid, self.hid),
            nn.ReLU()
        )
        self.pos_enc = PositionalEncoding(self.hid)

        # 2. 三类 attention
        self.transformer    = Transformer(self.hid, n_heads, num_layers=6, dropout=drop) # 历史到自身的叠层Transformer
        
        self.joint_transformer = Transformer(self.hid, n_heads, num_layers=6, dropout=drop)
        
        self.state2hist     = AttentionBlock(self.hid, n_heads, drop)  # 状态→历史

        # 3. state CNN encoder
        C, H, W = state_feature_shape
        self.state_extractor = nn.Sequential(
            ConvBlock(C, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 64),
            nn.Flatten(),
            nn.Linear(H * W * 64, self.hid),
        )

        # 4. actor / logits head
        self.fc = nn.Sequential(
            nn.Linear(self.hid, self.hid // 2),
            nn.ReLU(),
            nn.Linear(self.hid // 2, output_dim),
        )

    # ---------- forward ----------
    def forward(self, batch, hist_kpm:torch.Tensor=None):
        # ---- ① 历史序列 ----
        hist = self.embed(batch["history"])                   # (B,T,D)
        hist = hist + self.pos_enc(hist)                      # Pos‑enc
        hist = self.transformer(hist, key_padding_mask=hist_kpm)

        # ---- ② 当前手牌 ----
        state = self.state_extractor(batch["hand"])           # (B,D)
        state = state.unsqueeze(1)                            # ‑> (B,1,D)
        # ---- joint attention  ----
        joint = torch.cat([hist, state], dim=1)                # (B,T+1,D)
        
        mask_0 = torch.tensor([False] * hist.shape[0], dtype=torch.bool, device=hist.device).unsqueeze(1)  # (B,1)
        joint_kpm = torch.cat([mask_0, hist_kpm], dim=1) if hist_kpm is not None else None  # (B, T+1); if 语句缓解pylance报错
        
        joint = self.joint_transformer(joint,joint_kpm)
        state, hist = joint[:,0,:].unsqueeze(1), joint[:,1:,:]
        state = self.state2hist(state, k=hist,  v=hist, key_padding_mask=hist_kpm)       # Q=state

        # ---- ④ actor head ----
        logits = self.fc(state.squeeze(1))                    # (B,|A|)
        inf_mask = torch.clamp(torch.log(batch["action_mask"]), -1e38, 1e38)
        return logits + inf_mask
if __name__ == "__main__":
    # 测试代码
    model = TimeNet(95, (4, 4, 9), 235)
    model.to("cuda")
    
    # 创建一个随机的序列长度掩码，模拟变长序列
    batch_size = 32
    max_seq_len = 10
    # 为每个batch元素随机生成有效序列长度（1到max_seq_len之间）
    seq_lengths = torch.randint(1, max_seq_len, (batch_size,), device="cuda")
    # 创建掩码矩阵：True表示要被掩盖的位置（无效位置）
    mask = torch.arange(max_seq_len, device="cuda").expand(batch_size, max_seq_len) >= seq_lengths.unsqueeze(1)
    
    #print(seq_lengths,mask)
    x = {
        "history": torch.randn(32, max_seq_len, 95,device="cuda"),
        "hand": torch.randn(32, 4, 4, 9,device="cuda"),
        "action_mask": torch.ones(32, 235,device="cuda"),
    }
    output = model(x,mask)
    print(output.shape)  # 应该是 (32, 235)