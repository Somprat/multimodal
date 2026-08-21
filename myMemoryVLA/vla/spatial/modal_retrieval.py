from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import exp
from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. do cosine similarity between currnet observation and the memory in the same episode
# 2. do the 

# adding the per, cog and mem in the memory record
# might not need another memory bank. Just compare the feat of the current timestep with each of the one in the past
# and compre the position and time difference as well

        # self.bank[episode_id].append(BankEntry(
        #     timestep=timestep,
        #     feat=feat.detach().clone(),
        #     image_embedding=image_embedding,
        #     position=stored_position,
        #     task_tags=task_tags
        # ))


class RetrievalMode(str, Enum):
    DEFAULT = "default"
    SEMANTIC_SPATIAL_RECENT = "semantic_spatial_recent"
    OBJECT_STATE = "object_state"
    AUDIO_TEMPORAL_VISUAL = "audio_temporal_visual"
    NAVIGATION = "navigation"

@dataclass
class ModalMemoryRecord:
    id: str
    text: str = ""
    tokens: Optional[torch.Tensor] = None
    position: Optional[torch.Tensor] = None
    timestamp: Optional[float] = None
    # Sequence[str] = takes in a list/tuple of str
    # field() adds some special thing to that argumnet
    # in this case, default_factory=tuple means that if the value is not provided, it will initialized a blank tuple
    task_tags: Sequence[str] = field(default_factory=tuple)
    # we don't do Optional[Sequence][str].
    # That means if we don't provide the argument, it becomes None which is bad
    # we want it to be a blank tuple if it's not provided
    object_ids: Sequence[str] = field(default_factory=tuple)
    state: Mapping[str, Any] = field(default_factory=dict)
    modality: str = "unknown"


class QueryModeClassifier:
    """The CLIP + linear query router trained by train_query_router.py."""

    LABEL_TO_MODE = {
        "navigation": RetrievalMode.NAVIGATION,
        "object_state": RetrievalMode.OBJECT_STATE,
        "default": RetrievalMode.DEFAULT,
        "temporal": RetrievalMode.AUDIO_TEMPORAL_VISUAL,
    }

    def __init__(self, checkpoint_path: Optional[str] = None) -> None:
        from pathlib import Path

        from transformers import CLIPModel, CLIPTokenizer

        path = Path(checkpoint_path) if checkpoint_path else (
            Path(__file__).resolve().parents[2]
            / "checkpoints/query_router/clip_linear_4class.pt"
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        encoder_name = checkpoint["encoder"]

        self.tokenizer = CLIPTokenizer.from_pretrained(encoder_name)
        self.encoder = CLIPModel.from_pretrained(encoder_name).to(self.device).eval()
        self.classifier = nn.Linear(
            checkpoint["embedding_dim"], len(checkpoint["labels"])
        ).to(self.device)
        self.classifier.load_state_dict(checkpoint["state_dict"])
        self.classifier.eval()
        self.labels = checkpoint["labels"]

    @torch.inference_mode()
    def classify(self, query: ModalRetrievalQuery) -> Optional[RetrievalMode]:
        inputs = self.tokenizer(
            text=[query.text], padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        features = self.encoder.get_text_features(**inputs)
        embedding = features.pooler_output if hasattr(features, "pooler_output") else features
        embedding = F.normalize(embedding.float(), dim=-1)
        label = self.labels[self.classifier(embedding).argmax(dim=-1).item()]
        return self.LABEL_TO_MODE.get(label)



@dataclass
class ModalRetrievalQuery:
    text: str
    tokens: Optional[torch.Tensor] = None
    current_position: Optional[torch.Tensor] = None
    current_time: Optional[float] = None
    task_type: Optional[str] = None
    modality_hints: Sequence[str] = field(default_factory=tuple)
    object_ids: Sequence[str] = field(default_factory=tuple)



# doesn't allowed these weights to be changed
@dataclass(frozen=True)
class RetrievalWeights:
    semantic: float = 0.25
    spatial: float = 0.25
    temporal: float = 0.25
    task: float = 0.25


@dataclass
class RetrievalResult:
    # the info about each aspect: posistion, emebdding, texts, etc.
    memory: ModalMemoryRecord
    # retrieval score: higher = more relevant
    score: float
    breakdown: Mapping[str, float]
    # break down shows the various of info
    # {
#     "text_similarity": 0.72,
#     "spatial_match": 0.15,
#     "recency": 0.08
# }
    mode: RetrievalMode = None
    budget: str = "default"


TASK_MEMORY_BUDGETS = {
    "navigation": {
        "cog": 3,
        "per": 1,
        "spatial": 4,
    },
    "object_state": {
        "cog": 2,
        "per": 4,
        "spatial": 2,
    },
    "temporal": {
        "cog": 4,
        "per": 2,
        "spatial": 2,
    },
    "default": {
        "cog": 2,
        "per": 3,
        "spatial": 3,
    },
}

# do the cosine similarity of the the current tokens and the history here
def token_score(
        query: ModalRetrievalQuery,
        memory: ModalMemoryRecord
):
    score = F.cosine_similarity(query.tokens.float(), memory.tokens.float(), dim=-1).mean()
    return float(score.detach().cpu().item())



def spatial_score(
    query: ModalRetrievalQuery,
    memory: ModalMemoryRecord,
    spatial_scale: float = 2.0,
) -> float:
    if query.current_position is None or memory.position is None:
        return 0.0

    # do the same preprocessing
    query_position = query.current_position.float().flatten()
    memory_position = memory.position.float().flatten()
    if query_position.numel() != memory_position.numel():
        return 0.0

    # find the euclidean distance
    distance = torch.linalg.norm(query_position - memory_position).item()
    # the exp thing just mean small distance = higher score
    #  spatial scale control the influence of the distance on the score. more = less distance influence
    return exp(-distance / max(spatial_scale, 1e-6))



def temporal_score(
    query: ModalRetrievalQuery,
    memory: ModalMemoryRecord,
    temporal_scale: float = 60.0,
) -> float:
    if query.current_time is None or memory.timestamp is None:
        return 0.0

    # temporal score just find the difference
    age = abs(query.current_time - memory.timestamp)
    return exp(-age / max(temporal_scale, 1e-6))


def task_score(query: ModalRetrievalQuery, memory: ModalMemoryRecord) -> float:

    # Next step: implement the object Id
    score = 0.0
    # store stuff in set to prevent duplicates
    task_tags = {tag.lower() for tag in memory.task_tags}
    query_objects = {object_id.lower() for object_id in query.object_ids}
    memory_objects = {object_id.lower() for object_id in memory.object_ids}
    modality_hints = {hint.lower() for hint in query.modality_hints}

    # if the query's task type is in the memory task tags, task_score + 0.5
    if query.task_type is not None and query.task_type.lower() in task_tags:
        score += 0.5
    # similarly but for objects
    if query_objects and query_objects.intersection(memory_objects):
        score += 0.3
    # modality_hints are what type of memory would be useful for thiw query
    # task type are what we are trying to do: navigation, object_state. hinst are like visual, audio, ...
    if memory.modality.lower() in modality_hints:
        score += 0.2

    return _clamp01(score)




class ManualRetrievalRouter:
    """Select retrieval weights from explicit task metadata, with heuristic fallback."""

    # create dictionary pairs of the task's name and its optimal weights


    # store the key of the last dict as its value: easy for conversion from task_type to mode
    TASK_TYPE_TO_MODE = {
        "navigation": RetrievalMode.NAVIGATION,
        "navigate": RetrievalMode.NAVIGATION,
        "object_state": RetrievalMode.OBJECT_STATE,
        "state": RetrievalMode.OBJECT_STATE,
        "audio_temporal_visual": RetrievalMode.AUDIO_TEMPORAL_VISUAL,
        "audio_temporal": RetrievalMode.AUDIO_TEMPORAL_VISUAL,
        "semantic_spatial_recent": RetrievalMode.SEMANTIC_SPATIAL_RECENT,
        "recent": RetrievalMode.SEMANTIC_SPATIAL_RECENT,
    }

    NAVIGATION_TERMS = ("go to", "navigate", "where is", "find", "return to")
    RECENT_TERMS = ("last seen", "recent", "before", "earlier", "previously")
    STATE_TERMS = ("state", "open", "closed", "on", "off", "moved", "changed")

    MANUAL_TASK_LABELS = {
        "pick up the object and move it to a goal position.": "navigation",
        "pick up a designated object from a clutter of objects.": "object_state",
        "turn on the faucet by rotating a designated handle.": "default",
        "insert a designated object into the corresponding slot on a board.": "navigation",
        "plug the charger into the wall socket.": "navigation",
        "stack the red cube on top of the green cube.": "navigation",
        "insert the peg into the horizontal hole in a box.": "navigation",
        "pick up the red cube and move it to a goal position.": "navigation",
        "lift up the red cube by 0.2 meters.": "navigation"
    }

    # classify each query into different modes
    MODE_TO_BUDGET = {
        RetrievalMode.NAVIGATION: "navigation",
        RetrievalMode.OBJECT_STATE: "object_state",
        RetrievalMode.AUDIO_TEMPORAL_VISUAL: "temporal",
        RetrievalMode.SEMANTIC_SPATIAL_RECENT: "temporal",
        RetrievalMode.DEFAULT: "default",
    }

    def __init__(
        self,
        use_classifier: bool = False,
        classifier: Optional[QueryModeClassifier] = None,
    ) -> None:
        self.use_classifier = use_classifier
        self.classifier = classifier or (
            QueryModeClassifier() if use_classifier else None
        )

    # return the query's mode and its weight that we set up there
    def route(self, query: ModalRetrievalQuery) -> str:
        if self.use_classifier and self.classifier is not None:
            classified_mode = self.classifier.classify(query)
            if classified_mode is not None:
                return self.MODE_TO_BUDGET[classified_mode]
        


#        mode = self._mode_from_task_type(query) or self._mode_from_text(query.text)
        budget = self.MANUAL_TASK_LABELS.get(query.text, "default")
        return budget



    # both these functuons' end goals are modes
    def _mode_from_task_type(self, query: ModalRetrievalQuery) -> Optional[RetrievalMode]:
        if query.task_type is None:
            return None
        # input the query's task type (key of the TASK_TYPE_TO_MODE dictionary) to get the mode
        # Each mode is a dictionary that stores weight of the tasks
        return self.TASK_TYPE_TO_MODE.get(query.task_type.lower())

    def _mode_from_text(self, text: str) -> RetrievalMode:
        normalized = text.lower()

        # put in texts-> detect keyword in NAVIGATION_TERMS->return the mode
        if any(term in normalized for term in self.NAVIGATION_TERMS):
            return RetrievalMode.NAVIGATION
        if any(term in normalized for term in self.RECENT_TERMS):
            return RetrievalMode.SEMANTIC_SPATIAL_RECENT
        if any(term in normalized for term in self.STATE_TERMS):
            return RetrievalMode.OBJECT_STATE

        return RetrievalMode.DEFAULT


class ModalMemoryRetriever:
    def __init__(
        self,
        router: Optional[ManualRetrievalRouter] = None,
        use_classifier: bool = False,
        spatial_scale: float = 2.0,
        temporal_scale: float = 60.0,
    ) -> None:
        self.router = router or ManualRetrievalRouter(use_classifier=use_classifier)
        self.spatial_scale = spatial_scale
        self.temporal_scale = temporal_scale

    def retrieve(
        self,
        weights,
        query: ModalRetrievalQuery,
        memories: Sequence[ModalMemoryRecord],
        modal: str
    ) -> list[RetrievalResult]:

        budget_name = self.router.route(query)
        top_k = TASK_MEMORY_BUDGETS[budget_name][modal]
        
        results = []

        for memory in memories:
            token = token_score(query, memory)
            position = spatial_score(query, memory, spatial_scale=self.spatial_scale)
            time = temporal_score(query, memory, temporal_scale=self.temporal_scale)
            task = task_score(query, memory)
            # sum of weights*task score
            # task score was obtain by consine similarity of query and memory
            total = (
                weights["token"] * token
                + weights["position"] * position
                + weights["time"] * time
                + weights["task"] * task
            )
            # at the end, append memory, score, breakdown which is just some info and mode
            results.append(
                RetrievalResult(
                    memory=memory,
                    score=total,
                    breakdown={
                        "token": token,
                        "spatial": position,
                        "temporal": time,
                        "task": task,
                    },
                    budget=budget_name
                )
            )
        #sorting the result by scores which is the total variable with is the sum of ...
        results.sort(key=lambda result: result.score, reverse=True)
        # only return the top_k with the most total
        return results[:top_k]
    
def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))
