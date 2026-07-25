from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


def _load_retrieval_module():
    module_path = Path(__file__).with_name("retrieval.py")
    spec = importlib.util.spec_from_file_location("spatial_retrieval_test_module", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


retrieval = _load_retrieval_module()


def test_last_seen_mug_prefers_semantic_spatial_recent_match():
    query = retrieval.RetrievalQuery(
        text="Where did I last see the mug?",
        embedding=torch.tensor([1.0, 0.0, 0.0]),
        current_position=torch.tensor([0.0, 0.0, 0.0]),
        current_time=100.0,
        task_type="semantic_spatial_recent",
        object_ids=("mug",),
        modality_hints=("visual", "spatial"),
    )
    memories = [
        retrieval.MemoryRecord(
            id="recent_near_mug",
            embedding=torch.tensor([1.0, 0.0, 0.0]),
            position=torch.tensor([0.2, 0.0, 0.0]),
            timestamp=95.0,
            task_tags=("semantic_spatial_recent",),
            object_ids=("mug",),
            modality="visual",
        ),
        retrieval.MemoryRecord(
            id="old_near_mug",
            embedding=torch.tensor([1.0, 0.0, 0.0]),
            position=torch.tensor([0.2, 0.0, 0.0]),
            timestamp=10.0,
            task_tags=("semantic_spatial_recent",),
            object_ids=("mug",),
            modality="visual",
        ),
        retrieval.MemoryRecord(
            id="recent_near_plate",
            embedding=torch.tensor([0.0, 1.0, 0.0]),
            position=torch.tensor([0.1, 0.0, 0.0]),
            timestamp=99.0,
            task_tags=("semantic_spatial_recent",),
            object_ids=("plate",),
            modality="visual",
        ),
    ]

    results = retrieval.MemoryRetriever().retrieve(query, memories, top_k=3)

    assert results[0].memory.id == "recent_near_mug"
    assert results[0].mode == retrieval.RetrievalMode.SEMANTIC_SPATIAL_RECENT
    assert results[0].breakdown["semantic"] > 0.99
    assert results[0].breakdown["temporal"] > results[1].breakdown["temporal"]


def test_drawer_state_prefers_same_object_with_prior_state():
    query = retrieval.RetrievalQuery(
        text="Have I already opened this drawer?",
        embedding=torch.tensor([0.0, 1.0, 0.0]),
        current_position=torch.tensor([1.0, 0.0, 0.0]),
        current_time=50.0,
        task_type="object_state",
        object_ids=("drawer_1",),
        modality_hints=("visual",),
    )
    memories = [
        retrieval.MemoryRecord(
            id="drawer_1_open",
            embedding=torch.tensor([0.0, 1.0, 0.0]),
            position=torch.tensor([1.1, 0.0, 0.0]),
            timestamp=40.0,
            task_tags=("object_state",),
            object_ids=("drawer_1",),
            state={"open": True},
            modality="visual",
        ),
        retrieval.MemoryRecord(
            id="drawer_2_open",
            embedding=torch.tensor([0.0, 1.0, 0.0]),
            position=torch.tensor([1.1, 0.0, 0.0]),
            timestamp=49.0,
            task_tags=("object_state",),
            object_ids=("drawer_2",),
            state={"open": True},
            modality="visual",
        ),
        retrieval.MemoryRecord(
            id="drawer_1_unrelated_audio",
            embedding=torch.tensor([1.0, 0.0, 0.0]),
            position=torch.tensor([4.0, 0.0, 0.0]),
            timestamp=49.0,
            task_tags=("audio_temporal_visual",),
            object_ids=("drawer_1",),
            state={},
            modality="audio",
        ),
    ]

    results = retrieval.MemoryRetriever().retrieve(query, memories, top_k=3)

    assert results[0].memory.id == "drawer_1_open"
    assert results[0].memory.state["open"] is True
    assert results[0].mode == retrieval.RetrievalMode.OBJECT_STATE


def test_sound_query_prefers_near_time_audio_and_near_visual_observation():
    query = retrieval.RetrievalQuery(
        text="What made that sound?",
        embedding=torch.tensor([0.0, 0.0, 1.0]),
        current_position=torch.tensor([2.0, 0.0, 0.0]),
        current_time=200.0,
        task_type="audio_temporal_visual",
        modality_hints=("audio", "visual"),
    )
    memories = [
        retrieval.MemoryRecord(
            id="near_recent_audio",
            embedding=torch.tensor([0.0, 0.0, 1.0]),
            position=torch.tensor([2.1, 0.0, 0.0]),
            timestamp=199.0,
            task_tags=("audio_temporal_visual",),
            modality="audio",
        ),
        retrieval.MemoryRecord(
            id="near_recent_visual",
            embedding=torch.tensor([0.0, 0.0, 0.9]),
            position=torch.tensor([2.2, 0.0, 0.0]),
            timestamp=198.0,
            task_tags=("audio_temporal_visual",),
            modality="visual",
        ),
        retrieval.MemoryRecord(
            id="old_audio",
            embedding=torch.tensor([0.0, 0.0, 1.0]),
            position=torch.tensor([2.1, 0.0, 0.0]),
            timestamp=20.0,
            task_tags=("audio_temporal_visual",),
            modality="audio",
        ),
        retrieval.MemoryRecord(
            id="far_recent_audio",
            embedding=torch.tensor([0.0, 0.0, 1.0]),
            position=torch.tensor([20.0, 0.0, 0.0]),
            timestamp=199.0,
            task_tags=("audio_temporal_visual",),
            modality="audio",
        ),
    ]

    results = retrieval.MemoryRetriever().retrieve(query, memories, top_k=4)
    top_ids = [result.memory.id for result in results[:2]]

    assert results[0].memory.id == "near_recent_audio"
    assert "near_recent_visual" in top_ids
    assert results[0].mode == retrieval.RetrievalMode.AUDIO_TEMPORAL_VISUAL


def test_navigation_weights_spatial_relevance_more_than_recency():
    query = retrieval.RetrievalQuery(
        text="go to the drawer",
        embedding=torch.tensor([0.0, 1.0, 0.0]),
        current_position=torch.tensor([0.0, 0.0, 0.0]),
        current_time=100.0,
        object_ids=("drawer",),
        modality_hints=("visual", "spatial"),
    )
    memories = [
        retrieval.MemoryRecord(
            id="near_old_drawer",
            embedding=torch.tensor([0.0, 1.0, 0.0]),
            position=torch.tensor([0.1, 0.0, 0.0]),
            timestamp=0.0,
            task_tags=("navigation",),
            object_ids=("drawer",),
            modality="visual",
        ),
        retrieval.MemoryRecord(
            id="far_recent_drawer",
            embedding=torch.tensor([0.0, 1.0, 0.0]),
            position=torch.tensor([10.0, 0.0, 0.0]),
            timestamp=99.0,
            task_tags=("navigation",),
            object_ids=("drawer",),
            modality="visual",
        ),
    ]

    results = retrieval.MemoryRetriever().retrieve(query, memories, top_k=2)

    assert results[0].memory.id == "near_old_drawer"
    assert results[0].mode == retrieval.RetrievalMode.NAVIGATION
    assert results[0].breakdown["spatial"] > results[1].breakdown["spatial"]
    assert results[0].breakdown["temporal"] < results[1].breakdown["temporal"]
