import math
import mlx.core as mx
import mlx.nn as mxnn

from typing import Optional

class MultiHeadAttention(mxnn.Module):
    def __init__(
            self, 
            num_heads: int, 
            hidden_dim: int,
        ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.qkv = mxnn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.out_proj = mxnn.Linear(hidden_dim, hidden_dim, bias=False)
        self.rope = mxnn.RoPE(self.head_dim)

    def __call__(
            self, 
            x: mx.array, 
            mask: Optional[mx.array] = None, 
            cache: Optional[tuple[mx.array, mx.array]] = None
        ) -> tuple[mx.array, tuple[mx.array, mx.array]]:
        bsz, seq_len, hidden = x.shape
        
        q, k, v = self.qkv(x).split([hidden, 2 * hidden], axis=-1)

        q = q.reshape(bsz, seq_len, self.num_heads, -1).transpose(0, 2, 1, 3)
        k = k.reshape(bsz, seq_len, self.num_heads, -1).transpose(0, 2, 1, 3)
        v = v.reshape(bsz, seq_len, self.num_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            k_cache, v_cache = cache
            q = self.rope(q, offset=k_cache.shape[2])
            k = self.rope(k, offset=k_cache.shape[2])
            k = mx.concat([k_cache, k], axis=2)
            v = mx.concat([v_cache, v], axis=2)
        else:
            q = self.rope(q)
            k = self.rope(k)

        scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask

        weights = mx.softmax(scores, axis=-1)
        out = (weights @ v).transpose(0, 2, 1, 3).reshape(bsz, seq_len, -1)
        return self.out_proj(out), (k, v)


class FFN(mxnn.Module):
    def __init__(self, hidden_dim, mlp_hidden_dim):
        super().__init__()
        self.up1 = mxnn.Linear(hidden_dim, mlp_hidden_dim, bias=False)
        self.up2 = mxnn.Linear(hidden_dim, mlp_hidden_dim, bias=False)
        self.down = mxnn.Linear(mlp_hidden_dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down(mxnn.silu(self.up1(x)) * self.up2(x))


class DecoderBlock(mxnn.Module):
    def __init__(
            self, 
            hidden_dim: int, 
            mlp_hidden_dim: int, 
            num_heads: int
        ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.num_heads = num_heads

        self.attn = MultiHeadAttention(num_heads, hidden_dim)
        self.attn_norm = mxnn.RMSNorm(hidden_dim)
        
        self.ffn = FFN(hidden_dim, mlp_hidden_dim)
        self.ffn_norm = mxnn.RMSNorm(hidden_dim)

    def __call__(
        self, 
        x: mx.array, 
        mask: Optional[mx.array] = None, 
        cache: Optional[tuple[mx.array, mx.array]] = None
    ) -> tuple[mx.array, tuple[mx.array, mx.array]]:
        
        r, cache = (self.attn(self.attn_norm(x), mask, cache))
        x = x + r
        
        r = self.ffn(self.ffn_norm(x))
        x = x + r
        return x, cache

class GPTLlama(mxnn.Module):
    def __init__(self, vocab_size, num_layers, hidden_dim, mlp_hidden_dim, num_heads):
        super().__init__()
        self.vocab_size = vocab_size
        self.tok_embedding = mxnn.Embedding(vocab_size, hidden_dim)
        self.layers = [DecoderBlock(hidden_dim, mlp_hidden_dim, num_heads) 
                       for _ in range(num_layers)]
        self.norm = mxnn.RMSNorm(hidden_dim)
        self.cls_head = mxnn.Linear(hidden_dim, vocab_size, bias=False)

    @staticmethod
    def _create_additive_causal_mask(N: int, dtype: mx.Dtype = mx.float32) -> mx.array:
        indices = mx.arange(N)
        mask = indices[:, None] < indices[None]
        mask = mask.astype(dtype) * mx.finfo(dtype).min
        return mask
    
    def __call__(self, x: mx.array) -> mx.array:
        mask = self._create_additive_causal_mask(x.shape[1], dtype=self.tok_embedding.weight.dtype)

        x = self.tok_embedding(x)
        for l in self.layers:
            x, _ = l(x, mask)
        x = self.norm(x)
        return self.cls_head(x)

    def generate(self, x, temp=1.0):
        def sample(logits):
            if temp==0:
                return mx.argmax(logits, axis=-1)
            else:
                return mx.random.categorical(logits * 1 / temp)

        cache = []
        mask = self._create_additive_causal_mask(x.shape[1], dtype=self.tok_embedding.weight.dtype)

        x = self.tok_embedding(x)
        for l in self.layers:
            x, c = l(x, mask)
            cache.append(c)
        x = self.norm(x)
        out = self.cls_head(x[:, -1])
        out = sample(out)
        yield out

        while True:
            x = out[:, None]
            x = self.tok_embedding(x)
            for i in range(len(cache)):
                x, cache[i] = self.layers[i](x, mask=None, cache=cache[i])
            x = self.norm(x)
            out = self.cls_head(x[:, -1])
            out = sample(out)
            yield out