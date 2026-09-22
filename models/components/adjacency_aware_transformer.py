import torch 
import torch.nn as nn
import torch.nn.functional as F
from transformer import FeedForward

class AdjacencyAwareMultiHeadAttention(nn.Module):

    def __init__(self, embedding_dim, n_heads):
        super(AdjacencyAwareMultiHeadAttention, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.head_dim = embedding_dim // n_heads

        assert self.head_dim * n_heads == embedding_dim, 'Parameter embedding_dim must be divisible by n_heads.'

        self.w_q = nn.Linear(embedding_dim, embedding_dim)
        self.w_k = nn.Linear(embedding_dim, embedding_dim)
        self.w_v = nn.Linear(embedding_dim, embedding_dim)

        self.out_dense = nn.Linear(embedding_dim, embedding_dim)

        self.alpha = nn.Parameter(torch.randn(n_heads, 1, 1))
    
    def forward(self, x, adjacency):
        batch_size = x.size(0)

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        q = q.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

        adjacency_expand = adjacency.expand(self.n_heads, -1, -1)
        adjacency_expand *= self.alpha

        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attention_scores += adjacency_expand.expand(batch_size, -1, -1, -1)
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_out = torch.matmul(attention_probs, v).transpose(1, 2).contiguous().view(batch_size, -1, self.embedding_dim)

        return self.out_dense(attention_out)

class AdjacencyAwareTransformerBlock(nn.Module):

    def __init__(self, embedding_dim, n_heads, hidden_dim):
        super(AdjacencyAwareTransformerBlock, self).__init__()

        self.attention = AdjacencyAwareMultiHeadAttention(embedding_dim, n_heads)
        self.feed_forward = FeedForward(embedding_dim, hidden_dim)
        
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.norm4 = nn.LayerNorm(embedding_dim)
    
    def forward(self, x, adjacency):

        norm_x = self.norm1(x)
        attn_out = self.attention(norm_x, adjacency)
        attn_out = self.norm2(attn_out)
        out1 = x + attn_out

        norm_out1 = self.norm3(out1)
        ffn_out = self.feed_forward(norm_out1)
        ffn_out = self.norm4(ffn_out)
        out2 = out1 + ffn_out

        return out2

class AdjacencyAwareTransformer(nn.Module):

    def __init__(self, n_blocks, embedding_dim, n_heads, hidden_dim, adjacency):
        super(AdjacencyAwareTransformer, self).__init__()

        self.adjacency = adjacency

        self.blocks = nn.ModuleList(
            [
                AdjacencyAwareTransformerBlock(
                    embedding_dim,
                    n_heads,
                    hidden_dim
                ) for _ in range(n_blocks)
            ]
        )

        self.norm = nn.LayerNorm(embedding_dim)
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x, self.adjacency)
        return self.norm(x)