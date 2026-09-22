import torch 
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):

    def __init__(self, embedding_dim, n_heads):
        super(MultiHeadAttention, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.head_dim = embedding_dim // n_heads

        assert self.head_dim * n_heads == embedding_dim, 'Parameter embedding_dim must be divisible by n_heads.'

        self.w_q = nn.Linear(embedding_dim, embedding_dim)
        self.w_k = nn.Linear(embedding_dim, embedding_dim)
        self.w_v = nn.Linear(embedding_dim, embedding_dim)

        self.out_dense = nn.Linear(embedding_dim, embedding_dim)
    
    def forward(self, x, mask=None):
        batch_size = x.size(0)

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        q = q.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_out = torch.matmul(attention_probs, v).transpose(1, 2).contiguous().view(batch_size, -1, self.embedding_dim)

        return self.out_dense(attention_out)

class FeedForward(nn.Module):

    def __init__(self, embedding_dim, ffn_ratio):
        super(FeedForward, self).__init__()

        ffn_dim = int(ffn_ratio * embedding_dim)

        self.fc1 = nn.Linear(embedding_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embedding_dim)

        self.activation = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        return self.fc2(x)

class TransformerBlock(nn.Module):

    def __init__(self, embedding_dim, n_heads, ffn_ratio, init_std=0.02):
        super(TransformerBlock, self).__init__()

        self.attention = MultiHeadAttention(embedding_dim, n_heads)
        self.feed_forward = FeedForward(embedding_dim, ffn_ratio)
        
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.init_std = init_std
        self.apply(self._init_weights)
    
    def forward(self, x, mask=None):
        norm_x = self.norm1(x)
        attn_out = self.attention(norm_x, mask)
        out1 = x + attn_out

        norm_out1 = self.norm2(out1)
        ffn_out = self.feed_forward(norm_out1)
        out2 = out1 + ffn_out

        return out2

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=self.init_std)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0)

class Transformer(nn.Module):

    def __init__(self, n_blocks, embedding_dim, n_heads, ffn_ratio, init_std=0.02):
        super(Transformer, self).__init__()

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim,
                    n_heads,
                    ffn_ratio,
                    init_std
                ) for _ in range(n_blocks)
            ]
        )
    
    def forward(self, x, mask=None):
        for block in self.blocks:
            x = block(x, mask)
        return x