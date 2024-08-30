import flax.nnx as nnx
import orbax.checkpoint as ocp


class Test(nnx.Module):
    def __init__(self, dim, rngs: nnx.Rngs):
        self.layer1 = nnx.Linear(dim, dim, rngs=rngs)
        # self.layer2 = nnx.Dropout(0.1, rngs=rngs)
        self.layer3 = nnx.Linear(dim, dim, rngs=rngs)

    def __call__(self, x):
        x = self.layer3(self.layer1(x))
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
)

model = Test(10, nnx.Rngs(0))
checkpoint_manager.save(0, args=ocp.args.StandardSave(nnx.state(model)))
checkpoint_manager.wait_until_finished()

abs_model = nnx.eval_shape(lambda: nnx.state(Test(10, nnx.Rngs(0))))
restored = checkpoint_manager.restore(0, args=ocp.args.StandardRestore(abs_model))

print(restored.flat_state().keys())
