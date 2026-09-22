import torch 
import torch.nn as nn
import torch.nn.functional as F
from transformer import MultiHeadAttention, FeedForward

class LocalGlobalMultiHeadAttention(nn.Module):

    def __init__(self, embedding_dim, n_heads):
        super(LocalGlobalMultiHeadAttention, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.head_dim = embedding_dim // n_heads

        assert self.head_dim * n_heads == embedding_dim, 'Parameter embedding_dim must be divisible by n_heads.'

        self.w_q = nn.Linear(embedding_dim, embedding_dim)
        self.w_k = nn.Linear(embedding_dim, embedding_dim)
        self.w_v = nn.Linear(embedding_dim, embedding_dim)

        self.out_dense = nn.Linear(embedding_dim, embedding_dim)
    
    def forward(self, x, x_global):
        batch_size = x.size(0)

        q = self.w_q(x_global)
        k = self.w_k(x)
        v = self.w_v(x)

        q = q.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_out = torch.matmul(attention_probs, v).transpose(1, 2).contiguous().view(batch_size, -1, self.embedding_dim)

        return self.out_dense(attention_out)

class LocalGlobalTransformerBlock(nn.Module):

    def __init__(self, embedding_dim, n_heads, hidden_dim):
        super(LocalGlobalTransformerBlock, self).__init__()
        
        self.n_heads = n_heads

        self.lg_attention = LocalGlobalMultiHeadAttention(embedding_dim, n_heads)
        self.masked_attention = MultiHeadAttention(embedding_dim, n_heads)

        self.feed_forward = FeedForward(embedding_dim, hidden_dim)

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.norm4 = nn.LayerNorm(embedding_dim)
        self.norm5 = nn.LayerNorm(embedding_dim)
        self.norm6 = nn.LayerNorm(embedding_dim)
        self.norm7 = nn.LayerNorm(embedding_dim)

    def forward(self, x, x_global, mask):
        batch_size = x.size(0)
        mask = mask.expand(batch_size, self.n_heads, -1, -1)

        norm_x = self.norm1(x)
        masked_attn_out = self.masked_attention(norm_x, mask)
        masked_attn_out = self.norm2(masked_attn_out)
        out1 = x + masked_attn_out

        norm_out1 = self.norm3(out1)
        norm_x_global = self.norm7(x_global)
        lg_attn_out = self.lg_attention(norm_out1, norm_x_global)
        lg_attn_out = self.norm4(lg_attn_out)
        out2 = out1 + lg_attn_out

        norm_out2 = self.norm5(out2)
        ffn_out = self.feed_forward(norm_out2)
        ffn_out = self.norm6(ffn_out)
        out3 = out2 + ffn_out

        return out3

class GlobalTransformerBlock(nn.Module):

    def __init__(self, embedding_dim, n_heads, hidden_dim):
        super(GlobalTransformerBlock, self).__init__()

        self.attention = MultiHeadAttention(embedding_dim, n_heads)
        self.feed_forward = FeedForward(embedding_dim, hidden_dim)
        
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.norm4 = nn.LayerNorm(embedding_dim)
    
    def forward(self, x, mask=None):

        norm_x = self.norm1(x)
        attn_out = self.attention(norm_x, mask)
        attn_out = self.norm2(attn_out)
        out1 = x + attn_out

        norm_out1 = self.norm3(out1)
        ffn_out = self.feed_forward(norm_out1)
        ffn_out = self.norm4(ffn_out)
        out2 = out1 + ffn_out

        return out2

class LocalGlobalTransformer(nn.Module):

    def __init__(self, n_local_blocks, n_global_blocks, embedding_dim, n_heads, hidden_dim, adjacency):
        super(LocalGlobalTransformer, self).__init__()
        
        self.adjacency = adjacency

        self.global_blocks = nn.ModuleList(
            [
                GlobalTransformerBlock(
                    embedding_dim,
                    n_heads,
                    hidden_dim
                ) for _ in range(n_global_blocks)
            ]
        )

        self.local_global_blocks = nn.ModuleList(
            [
                LocalGlobalTransformerBlock(
                    embedding_dim,
                    n_heads,
                    hidden_dim
                ) for _ in range(n_local_blocks)
            ]
        )

        self.norm = nn.LayerNorm(embedding_dim)
    
    def forward(self, x):
        input_ = x

        for block in self.global_blocks:
            x = block(x, mask=None)
        
        x_lg = self.local_global_blocks[0](input_, x, self.adjacency)
        for block in self.local_global_blocks[1:]:
            x_lg = block(x_lg, x, self.adjacency)
        return self.norm(x_lg)