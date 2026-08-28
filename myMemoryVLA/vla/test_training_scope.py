import torch.nn as nn

from vla.memory_vla import MemoryVLA


class _DummyVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 2)
        self.trainable_module_keys = ["proj"]


def _make_model(*, freeze_vlm: bool, freeze_action_model: bool) -> MemoryVLA:
    model = MemoryVLA.__new__(MemoryVLA)
    nn.Module.__init__(model)
    model.vlm = _DummyVLM()
    model.cog_mem_bank = nn.Linear(2, 2)
    model.per_mem_bank = nn.Linear(2, 2)
    model.spatial_mem_bank = nn.Linear(2, 2)
    model.spatial_to_per_fusion = nn.Linear(2, 2)
    model.episodic_bank = nn.Linear(2, 2)
    model.action_model = nn.Linear(2, 2)
    model.freeze_vlm = freeze_vlm
    model.freeze_action_model = freeze_action_model
    return model


def _is_trainable(module: nn.Module) -> bool:
    return any(parameter.requires_grad for parameter in module.parameters())


def test_freeze_vlm_also_freezes_perceptual_and_cognitive_memory() -> None:
    model = _make_model(freeze_vlm=True, freeze_action_model=False)
    model.apply_training_scope()

    assert not _is_trainable(model.vlm)
    assert not _is_trainable(model.cog_mem_bank)
    assert not _is_trainable(model.per_mem_bank)
    assert _is_trainable(model.spatial_mem_bank)
    assert _is_trainable(model.spatial_to_per_fusion)
    assert _is_trainable(model.episodic_bank)
    assert _is_trainable(model.action_model)


def test_freeze_action_model_is_independent_of_vlm_scope() -> None:
    model = _make_model(freeze_vlm=True, freeze_action_model=True)
    model.apply_training_scope()

    assert not _is_trainable(model.action_model)
    assert not model.action_model.training
    assert _is_trainable(model.spatial_mem_bank)
    assert _is_trainable(model.episodic_bank)
