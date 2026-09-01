import torch

from vla.episodic.episodic_bank import BankEntry, EpisodicMemBank, MemoryUnit
from vla.memory_vla import GateFusion


def _entry(value: float) -> BankEntry:
    return BankEntry(
        timestep=torch.tensor(0),
        feat=torch.full((1, 2), value),
        image_embedding=torch.full((2,), value),
        task_tags=("test",),
    )


def _unit(*, score: float, success: bool, complete: bool = True) -> MemoryUnit:
    entry = _entry(score) if complete else None
    return MemoryUnit(
        instruction_embedding=torch.tensor([score]),
        scene_embedding=torch.tensor([score]),
        success=success,
        cog_mem_bank=entry,
        per_mem_bank=entry,
    )


def test_new_fusion_gate_initially_preserves_pretrained_input() -> None:
    gate = GateFusion(dim=4, preserve_first_input=True)
    pretrained = torch.randn(2, 3, 4)
    new_context = torch.randn(2, 3, 4) * 100

    fused = gate(pretrained, new_context)
    initial_scale = torch.sigmoid(torch.tensor(6.0))
    expected = initial_scale * pretrained + (1 - initial_scale) * new_context

    assert initial_scale > 0.997
    assert torch.allclose(fused, expected)


def test_episodic_retrieval_excludes_failures_and_incomplete_rollouts() -> None:
    bank = EpisodicMemBank.__new__(EpisodicMemBank)
    torch.nn.Module.__init__(bank)
    bank.top_k = 2
    bank.bank = {
        1: _unit(score=0.5, success=True),
        2: _unit(score=1.0, success=False),
        3: _unit(score=0.9, success=True, complete=False),
    }
    bank.instruction_score = lambda memory, _: memory.instruction_embedding.item()
    bank.image_score = lambda memory, _: memory.scene_embedding.item()

    selected = bank.retrieve(current_instruction=["test"], initial_frame=[object()])

    assert selected == [bank.bank[1]]


def test_failed_episode_is_removed_instead_of_consuming_capacity() -> None:
    bank = EpisodicMemBank.__new__(EpisodicMemBank)
    torch.nn.Module.__init__(bank)
    bank.episode_id = 2
    bank.bank = {1: _unit(score=0.5, success=True, complete=False)}

    bank.end_episode(False, [_entry(0.5)], [_entry(0.5)])

    assert bank.bank == {}


def test_episode_completion_uses_the_explicit_episode_id() -> None:
    bank = EpisodicMemBank.__new__(EpisodicMemBank)
    torch.nn.Module.__init__(bank)
    bank.episode_id = 3
    bank.bank = {
        1: _unit(score=0.1, success=True, complete=False),
        2: _unit(score=0.2, success=True, complete=False),
    }

    bank.end_episode(
        True,
        [_entry(0.7)],
        [_entry(0.8)],
        episode_id=1,
    )

    assert bank.bank[1].cog_mem_bank is not None
    assert bank.bank[1].per_mem_bank is not None
    assert bank.bank[2].cog_mem_bank is None
    assert bank.bank[2].per_mem_bank is None
