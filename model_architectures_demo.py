"""
模型架构三大核心组件教学 Demo
================================
面向深度学习新手，逐行讲解：
  1. Scaled Dot-Product Attention（缩放点积注意力）
  2. Multi-Head Attention（多头注意力，MHA）
  3. MLA — Multi-Head Latent Attention（DeepSeek-V2 提出的潜在注意力）
  4. MoE 专家路由（Mixture of Experts，Top-K 门控 + token 分发）
  5. SwiGLU FFN（作为 MoE 的专家网络）

运行：  python model_architectures_demo.py
依赖：  torch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)  # 固定随机种子，方便复现


# ============================================================
# 第一部分：Scaled Dot-Product Attention
# ============================================================
def scaled_dot_product_attention(
    Q: torch.Tensor,        # 查询    shape: (batch, heads, seq_q, head_dim)
    K: torch.Tensor,        # 键      shape: (batch, heads, seq_k, head_dim)
    V: torch.Tensor,        # 值      shape: (batch, heads, seq_k, head_dim)
    mask: torch.Tensor | None = None,  # 掩码 shape: (1, 1, seq_q, seq_k) 或可广播
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    公式：  Attention(Q,K,V) = softmax( Q·K^T / sqrt(d_k) ) · V

    核心思想：每个 query 去和所有 key 做"相似度打分"，得分越高，对应的 value 被加权越多。
    """
    d_k = Q.size(-1)  # head_dim，缩放因子，防止点积过大导致 softmax 饱和

    # 第一步：Q·K^T  得到相似度分数矩阵
    # (batch, heads, seq_q, head_dim) @ (batch, heads, head_dim, seq_k)
    #        -> (batch, heads, seq_q, seq_k)
    # 即：第 i 个 query 对第 j 个 key 的打分为 scores[..., i, j]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)

    # 第二步：可选掩码（例如因果掩码，让第 i 个位置看不到未来的 key）
    if mask is not None:
        # mask 中为 True 的位置表示"不允许关注"，将其分数设为 -inf
        scores = scores.masked_fill(mask, float("-inf"))

    # 第三步：softmax 把分数变成 (0,1) 之间、每行和为 1 的权重
    # (batch, heads, seq_q, seq_k)
    attn_weights = torch.softmax(scores, dim=-1)

    # 第四步：用权重对 V 加权求和，得到最终输出
    # (batch, heads, seq_q, seq_k) @ (batch, heads, seq_k, head_dim)
    #        -> (batch, heads, seq_q, head_dim)
    output = torch.matmul(attn_weights, V)

    return output, attn_weights


# ============================================================
# 第二部分：Multi-Head Attention（MHA）
# ============================================================
class MultiHeadAttention(nn.Module):
    """
    多头注意力：把 Q/K/V 投影后"切"成多个头，每个头独立做注意力，最后再拼回来。
    好处：不同头可以学到不同的注意力模式（语法、语义、指代等）。
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads  # 每个头的维度

        # 三个线性投影：把输入分别变成 Q、K、V
        # 这里用一个大矩阵一次性投影，再切分，等价于三个独立 Linear
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # 把多头拼接后的输出再投影回 d_model
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        # x shape: (batch, seq_len, d_model)
        B, T, C = x.shape

        # 第一步：线性投影得到 Q, K, V
        # 每个还是 (B, T, d_model)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # 第二步：reshape 成多头形式
        # (B, T, d_model) -> (B, T, num_heads, head_dim) -> (B, num_heads, T, head_dim)
        # 这样每个头可以独立做注意力
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # 第三步：调用缩放点积注意力
        # attn_out: (B, num_heads, T, head_dim)
        attn_out, _ = scaled_dot_product_attention(Q, K, V, mask)

        # 第四步：把多头拼接回来
        # (B, num_heads, T, head_dim) -> (B, T, num_heads, head_dim) -> (B, T, d_model)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)

        # 第五步：输出投影
        out = self.out_proj(attn_out)  # (B, T, d_model)
        return out


# ============================================================
# 第三部分：MLA — Multi-Head Latent Attention
# ============================================================
# MLA 是 DeepSeek-V2 提出的优化注意力。核心思想：
#   - 普通 MHA 中 K 和 V 占用了大量 KV Cache（显存瓶颈）。
#   - MLA 把 K/V 先压缩到一个"低维潜在空间"再缓存，推理时再展开，
#     从而大幅减少 KV Cache 显存。
# 这里实现的是教学版，体现核心思路：低秩压缩 KV。
class MLA(nn.Module):
    """
    教学版 MLA：
      - Q 仍然投影到 (num_heads, head_dim) 高维空间
      - K/V 先压缩到一个低维 latent_dim，缓存这个低维向量；
        需要做注意力时，再把它上投影回 K/V 的维度。
    这样 KV Cache 从 seq_len * (2 * d_model) 降到 seq_len * latent_dim。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        kv_latent_dim: int,   # KV 的低维潜在维度，通常 << d_model
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.kv_latent_dim = kv_latent_dim

        # Q 正常投影
        self.q_proj = nn.Linear(d_model, d_model)

        # KV 压缩：把原始 d_model 压到 kv_latent_dim（这就是要缓存的东西）
        self.kv_down_proj = nn.Linear(d_model, kv_latent_dim, bias=False)

        # 从潜在空间恢复出 K 和 V（共享输入，各自一个上投影）
        self.k_up_proj = nn.Linear(kv_latent_dim, d_model, bias=False)
        self.v_up_proj = nn.Linear(kv_latent_dim, d_model, bias=False)

        self.out_proj = nn.Linear(d_model, d_model)

    def compress_kv(self, x: torch.Tensor) -> torch.Tensor:
        """把输入压缩成低维 KV latent（这才是被缓存的对象）。"""
        # x: (B, T, d_model) -> kv_latent: (B, T, kv_latent_dim)
        return self.kv_down_proj(x)

    def forward(self, x: torch.Tensor, mask=None):
        B, T, C = x.shape

        # --- Q 分支：正常高维投影 ---
        Q = self.q_proj(x)                                  # (B, T, d_model)
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        # Q: (B, num_heads, T, head_dim)

        # --- KV 分支：先压缩再展开 ---
        kv_latent = self.compress_kv(x)                     # (B, T, kv_latent_dim)
        K = self.k_up_proj(kv_latent)                       # (B, T, d_model)
        V = self.v_up_proj(kv_latent)                       # (B, T, d_model)

        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        # K, V: (B, num_heads, T, head_dim)

        # --- 标准点积注意力 ---
        attn_out, _ = scaled_dot_product_attention(Q, K, V, mask)  # (B, num_heads, T, head_dim)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(attn_out)


# ============================================================
# 第四部分：SwiGLU FFN（作为 MoE 的"专家"网络）
# ============================================================
class SwiGLU(nn.Module):
    """
    SwiGLU = SiLU(xW1) * (xW2)  然后  再乘 W3
    相比 ReLU FFN，SwiGLU 在大模型中表现更好，是当前主流选择。
    """

    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)   # 门控分支
        self.w2 = nn.Linear(d_model, hidden_dim, bias=False)   # 值分支
        self.w3 = nn.Linear(hidden_dim, d_model, bias=False)   # 输出投影

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_model)
        gate = F.silu(self.w1(x))      # (..., hidden_dim)
        value = self.w2(x)             # (..., hidden_dim)
        return self.w3(gate * value)   # (..., d_model)


# ============================================================
# 第五部分：MoE 专家路由（Mixture of Experts）
# ============================================================
class MoE(nn.Module):
    """
    MoE 核心思想：
      - 有 N 个并行的"专家"网络（每个是一个 SwiGLU FFN）。
      - 一个"门控（router/gate）"网络根据输入 token 决定：
        把这个 token 送给哪 top-k 个专家处理，并以多大权重合并结果。
      - 这样总参数量可以很大，但每个 token 只激活少数专家，
        计算量 ≈ 普通 FFN，却能容纳更多知识。
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,       # 专家总数，例如 8
        num_experts_per_tok: int,  # 每个 token 激活几个专家，例如 2（top-k）
        hidden_dim: int,        # 每个专家内部隐藏维度
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = num_experts_per_tok

        # 门控：把 token 投影到 num_experts 维，得到对每个专家的打分
        self.gate = nn.Linear(d_model, num_experts, bias=False)

        # 所有专家。用 ModuleList 装起来，每个专家独立参数。
        self.experts = nn.ModuleList([
            SwiGLU(d_model, hidden_dim) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        B, T, C = x.shape

        # 为了逐 token 路由，先展平成 (B*T, C)
        x_flat = x.view(-1, C)                     # (N, C), N = B*T
        N = x_flat.shape[0]

        # 第一步：门控打分
        # scores: (N, num_experts)  每个 token 对每个专家的原始打分
        scores = self.gate(x_flat)

        # 第二步：取 top-k 个专家的索引和对应打分
        # topk_indices: (N, top_k)    topk_weights: (N, top_k)
        topk_weights, topk_indices = torch.topk(scores, self.top_k, dim=-1)

        # 第三步：把 top-k 的权重做 softmax，变成和为 1 的混合系数
        topk_weights = torch.softmax(topk_weights, dim=-1)  # (N, top_k)

        # 第四步：逐专家分发 token 并计算，再按权重合并
        # 初始化输出 (N, C)
        y = torch.zeros_like(x_flat)

        for expert_idx in range(self.num_experts):
            # 找出"哪些 token 被分配给了这个专家"
            # token_mask: (N, top_k) 的布尔矩阵，True 表示该位置是当前专家
            token_mask = (topk_indices == expert_idx)        # (N, top_k)
            if not token_mask.any():
                continue  # 这个专家没被选中，跳过

            # 对每个选中的位置（token_i, slot_j），取对应权重并计算
            # 这里用一个简化但清晰的循环实现教学
            for token_i in range(N):
                for slot_j in range(self.top_k):
                    if topk_indices[token_i, slot_j] == expert_idx:
                        w = topk_weights[token_i, slot_j]            # 标量权重
                        expert_out = self.experts[expert_idx](x_flat[token_i:token_i+1])  # (1, C)
                        y[token_i] += w * expert_out.squeeze(0)

        # 恢复成 (B, T, C)
        return y.view(B, T, C)


# ============================================================
# 第六部分：把它们组装成一个 Transformer Block
# ============================================================
class TransformerBlock(nn.Module):
    """一个完整的 Transformer Block：Attention + MoE FFN，带残差和 RMSNorm。"""

    def __init__(self, d_model, num_heads, num_experts, top_k, ffn_hidden, kv_latent_dim=None):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        # 可选：用 MLA 替换 MHA；这里演示 MHA，下方会单独测 MLA
        self.attn = MultiHeadAttention(d_model, num_heads)

        self.norm2 = nn.RMSNorm(d_model)
        self.moe = MoE(d_model, num_experts, top_k, ffn_hidden)

    def forward(self, x, mask=None):
        # 残差连接 1：注意力
        x = x + self.attn(self.norm1(x), mask)
        # 残差连接 2：MoE 前馈
        x = x + self.moe(self.norm2(x))
        return x


# ============================================================
# 演示主函数
# ============================================================
def main():
    print("=" * 70)
    print("模型架构三大核心组件教学 Demo")
    print("=" * 70)

    B, T, C = 2, 5, 64          # batch=2, seq_len=5, d_model=64
    num_heads = 4
    head_dim = C // num_heads   # 16

    x = torch.randn(B, T, C)
    print(f"\n[输入] x shape = {tuple(x.shape)}  (B={B}, T={T}, C={C})")

    # ---------------- 1. Scaled Dot-Product Attention ----------------
    print("\n" + "-" * 70)
    print("1) Scaled Dot-Product Attention")
    print("-" * 70)
    Q = torch.randn(B, num_heads, T, head_dim)
    K = torch.randn(B, num_heads, T, head_dim)
    V = torch.randn(B, num_heads, T, head_dim)
    out, weights = scaled_dot_product_attention(Q, K, V)
    print(f"   Q/K/V shape = {tuple(Q.shape)}")
    print(f"   attn_weights shape = {tuple(weights.shape)}  (每个 query 对所有 key 的权重)")
    print(f"   output shape = {tuple(out.shape)}")

    # 因果掩码演示：让第 i 个位置看不到 i 之后的位置
    causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1).unsqueeze(0).unsqueeze(0)
    out_c, w_c = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
    # 验证：第一行只关注第 0 列，其余权重应为 0
    print(f"   [因果掩码验证] w_c[0,0,0] (第0个query的注意力) = {w_c[0,0,0].tolist()}")
    print(f"   [因果掩码验证] w_c[0,0,1] (第1个query, 只能看0,1) = {w_c[0,0,1].tolist()}")

    # ---------------- 2. Multi-Head Attention ----------------
    print("\n" + "-" * 70)
    print("2) Multi-Head Attention")
    print("-" * 70)
    mha = MultiHeadAttention(d_model=C, num_heads=num_heads)
    out_mha = mha(x)
    print(f"   输入 x: {tuple(x.shape)}  ->  输出: {tuple(out_mha.shape)}")
    print(f"   参数量: {sum(p.numel() for p in mha.parameters())}")

    # ---------------- 3. MLA ----------------
    print("\n" + "-" * 70)
    print("3) MLA (Multi-Head Latent Attention)")
    print("-" * 70)
    kv_latent = 16   # 远小于 d_model=64
    mla = MLA(d_model=C, num_heads=num_heads, kv_latent_dim=kv_latent)
    out_mla = mla(x)
    kv_cache = mla.compress_kv(x)
    print(f"   输入 x: {tuple(x.shape)}  ->  输出: {tuple(out_mla.shape)}")
    print(f"   KV Cache 压缩后: {tuple(kv_cache.shape)}  "
          f"(节省比例: {1 - kv_latent / (2 * C):.0%})")

    # ---------------- 4. MoE 专家路由 ----------------
    print("\n" + "-" * 70)
    print("4) MoE 专家路由 (8 experts, top-2)")
    print("-" * 70)
    moe = MoE(d_model=C, num_experts=8, num_experts_per_tok=2, hidden_dim=128)
    out_moe = moe(x)
    print(f"   输入 x: {tuple(x.shape)}  ->  输出: {tuple(out_moe.shape)}")
    # 统计每个 token 选中了哪些专家
    scores = moe.gate(x.view(-1, C))
    topk_w, topk_idx = torch.topk(scores, 2, dim=-1)
    print(f"   每个 token 的 top-2 专家索引:\n{topk_idx}")
    print(f"   总参数量: {sum(p.numel() for p in moe.parameters())}  "
          f"(= 8 个专家 + 1 个门控)")

    # ---------------- 5. 组装的 Transformer Block ----------------
    print("\n" + "-" * 70)
    print("5) 完整 Transformer Block (MHA + MoE)")
    print("-" * 70)
    block = TransformerBlock(
        d_model=C, num_heads=num_heads,
        num_experts=8, top_k=2, ffn_hidden=128,
    )
    out_block = block(x)
    print(f"   输入 x: {tuple(x.shape)}  ->  输出: {tuple(out_block.shape)}")

    print("\n" + "=" * 70)
    print("全部演示完成！请结合上方代码注释理解每一步。")
    print("=" * 70)


if __name__ == "__main__":
    main()
