import flax.nnx as nnx
import jax.numpy as jnp
import orbax.checkpoint as ocp


class Test(nnx.Module):
    def __init__(self, dim, rngs: nnx.Rngs):
        self.layer1 = nnx.Linear(dim, dim, rngs=rngs)
        self.layer2 = nnx.Dropout(0.1, rngs=rngs)
        self.layer3 = nnx.Linear(dim, dim, rngs=rngs)

    def __call__(self, x):
        x = self.layer3(self.layer2(self.layer1(x)))
        return x


checkpoint_manager = ocp.CheckpointManager(
    ocp.test_utils.erase_and_create_empty(
        "/var/local/ML/TRAIN/STAGE/troubleshoot_checkpointing/"
    ),
    options=ocp.CheckpointManagerOptions(
        max_to_keep=2,
        keep_checkpoints_without_metrics=False,
        create=True,
    ),
    item_names=("state", "layer2_dropout_key"),
)

model = Test(10, nnx.Rngs(0))
model_state = nnx.state(model).flat_state()
layer2_dropout_key = model_state[("layer2", "rngs", "default", "key")].value

# The Dropout layers RNG key had to be replaced with a dummy to
# allow checkpoint saving
# Error seen
# TypeError: Cannot interpret 'key<fry>' as a data type
model_state[("layer2", "rngs", "default", "key")] = nnx.VariableState(
    type=nnx.Param, value=jnp.array([])
)

# The RNG key had to be saved with its special checkpointer
checkpoint_manager.save(
    0,
    args=ocp.args.Composite(
        state=ocp.args.StandardSave(nnx.State.from_flat_path(model_state)),
        layer2_dropout_key=ocp.args.JaxRandomKeySave(layer2_dropout_key),
    ),
)
checkpoint_manager.wait_until_finished()

abs_model = nnx.eval_shape(lambda: nnx.State.from_flat_path(model_state))

# Checkpoint restoration also does not work
# The two items have to be restored separately
restored = checkpoint_manager.restore(
    0,
    args=ocp.args.Composite(
        state=ocp.args.StandardRestore(abs_model),
    ),
)

restored_key = checkpoint_manager.restore(
    0,
    args=ocp.args.Composite(
        # state=ocp.args.StandardRestore(abs_model),
        layer2_dropout_key=ocp.args.JaxRandomKeyRestore(),
    ),
)

# Model restoration is equally not straightforward
restored_model_state = restored["state"].flat_state()
restored_model_state[("layer2", "rngs", "default", "key")] = nnx.VariableState(
    type=nnx.Param, value=restored_key["layer2_dropout_key"]
)

abs_graph_def, abs_state = nnx.split(Test(10, nnx.Rngs(0)))
restored_model = nnx.merge(
    abs_graph_def, nnx.State.from_flat_path(restored_model_state)
)

print(nnx.state(restored_model).flat_state().keys())
