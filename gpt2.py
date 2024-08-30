from dataclasses import dataclass

import flax.nnx as nn
import jax
import jax.numpy as jnp


@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50257  # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12  # number of layers
    n_head: int = 12  # number of heads
    n_embd: int = 768  # embedding dimension
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig, rngs: nn.Rngs):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, rngs=rngs)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, rngs=rngs)
        #
        self.n_head = config.n_head
        self.dropout = config.dropout
        self.rngs = rngs

    def __call__(self, x: jax.Array):
        def attn(x):
            B, T, C = (
                x.shape
            )  # batch size, sequence length, embedding dimensionality (n_embd)
            # calculate query, key, values for all heads in batch and move head forward to be the batch dim
            # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
            # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
            qkv = self.c_attn(x)
            q, k, v = jnp.array_split(qkv, 3, axis=2)
            q = q.reshape(B, T, self.n_head, C // self.n_head)
            k = k.reshape(B, T, self.n_head, C // self.n_head)
            v = v.reshape(B, T, self.n_head, C // self.n_head)
            # mask=jnp.tril(jnp.ones((T, T))).reshape((1, 1, T, T))
            mask = nn.make_causal_mask(jnp.ones((B, T)))
            y = nn.dot_product_attention(
                q,
                k,
                v,
                mask=mask,
                dropout_rate=self.dropout,
                dropout_rng=self.rngs.default.key.value,
            )  # attention
            y = y.reshape(B, T, C)
            return y

        # output projection
        y = self.c_proj(attn(x))
        return y


class MLP(nn.Module):
    def __init__(self, config: GPTConfig, rngs: nn.Rngs):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, rngs=rngs)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, rngs=rngs)
        self.mlp_dropout = nn.Dropout(config.dropout, rngs=rngs)

    def __call__(self, x: jax.Array):
        x = self.c_fc(x)
        x = nn.gelu(x, approximate=True)
        x = self.c_proj(x)
        x = self.mlp_dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config: GPTConfig, rngs: nn.Rngs):
        super().__init__()
        self.ln_1 = nn.LayerNorm(
            config.n_embd, epsilon=1e-5, use_fast_variance=False, rngs=rngs
        )  # 2
        self.ln_2 = nn.LayerNorm(
            config.n_embd, epsilon=1e-5, use_fast_variance=False, rngs=rngs
        )  # 2
        self.attn = CausalSelfAttention(config, rngs=rngs)  # 4
        self.mlp = MLP(config, rngs=rngs)  # 4

    def __call__(self, x: jax.Array):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig, rngs: nn.Rngs):
        super().__init__()
        self.config = config
        self.wte = nn.Embed(config.vocab_size, config.n_embd, rngs=rngs)  # 1
        self.wpe = nn.Embed(config.block_size, config.n_embd, rngs=rngs)  # 1
        self.embedding_dropout = nn.Dropout(config.dropout, rngs=rngs)
        self.h = [
            Block(config, rngs=rngs) for _ in range(config.n_layer)
        ]  # 12 * 12 = 144
        self.ln_f = nn.LayerNorm(
            config.n_embd, epsilon=1e-5, use_fast_variance=False, rngs=rngs
        )  # 2
        self.lm_head = nn.Linear(
            config.n_embd, config.vocab_size, use_bias=False, rngs=rngs
        )  # 2
        # Total: 149

    def __call__(self, idx: jax.Array):
        # idx is of shape (B, T)
        B, T = idx.shape
        assert (
            T <= self.config.block_size
        ), f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = jnp.arange(0, T)  # shape (T)
        pos_emb = self.wpe(pos)  # position embeddings of shape (T, n_embd)
        tok_emb = self.wte(idx)  # token embeddings of shape (B, T, n_embd)
        x = self.embedding_dropout(tok_emb + pos_emb)
        # forward the blocks of the transformer
        for block in self.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return logits

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import FlaxGPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config, nn.Rngs(0))

        # init a huggingface/transformers model
        model_hf = FlaxGPT2LMHeadModel.from_pretrained(model_type)
        from flax.core import unfreeze
        from flax.traverse_util import flatten_dict

        jax_modules_dict = {}
        for module_pair in model.iter_modules():
            if type(module_pair[1]).__name__ in [
                "Block",
                "CausalSelfAttention",
                "GPT",
                "MLP",
            ]:
                continue
            module_path = ".".join([str(x) for x in module_pair[0]])
            module = module_pair[1]
            jax_modules_dict[module_path] = module

        print(f"Length of prepared JAX modules dict: {len(jax_modules_dict)}")
        # As Embed has only one matrix and lm_head has no bias
        # (len(jax_modules_dict) *2) - 3 should ne equal to Pytorch state dict length

        params = unfreeze(model_hf.params["transformer"])
        params = flatten_dict(params, sep=".")

        for param in params:
            t = param.split(".")[-1]  # Inner key Eg. wpe.embedding
            jax_key = ".".join(param.split(".")[:-1:])
            if params[param].shape == jax_modules_dict[jax_key].__dict__[t].value.shape:
                if "c_proj" in param:
                    # For this alone in terms of dimensions no transpose was needed 768 x 768
                    # and ended up causing a bug
                    jax_modules_dict[jax_key].__dict__[t].value = jnp.copy(
                        params[param].T
                    )
                else:
                    jax_modules_dict[jax_key].__dict__[t].value = jnp.copy(
                        params[param]
                    )
            elif (
                params[param].T.shape
                == jax_modules_dict[jax_key].__dict__[t].value.shape
            ):
                # print("Transposing ", param)
                jax_modules_dict[jax_key].__dict__[t].value = jnp.copy(params[param].T)
            else:
                print(f"Shape mismatch for {param}")

        jax_modules_dict["lm_head"].kernel.value = jnp.copy(params["wte.embedding"].T)

        return model
