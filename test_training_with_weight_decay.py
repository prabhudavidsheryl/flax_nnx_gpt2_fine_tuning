import flax.nnx as nnx
import optax

from gpt2 import GPT

model = GPT.from_pretrained("gpt2")


def model_state_decay_mask(model: GPT):
    flat_state = nnx.state(model).flat_state()
    flat_mask = flat_state.copy()
    for key in flat_state.keys():
        if "dropout" in key[0]:
            continue
        flat_mask[key] = key[-1] not in ("bias", "embedding", "scale", "count", "key")
    return nnx.State.from_flat_path(flat_mask)


model = GPT.from_pretrained("gpt2")
model.train()
mask = model_state_decay_mask(model)

tx = optax.adamw(learning_rate=1e-4, weight_decay=1e-4, mask=mask)
optimizer = nnx.Optimizer(model, tx)
