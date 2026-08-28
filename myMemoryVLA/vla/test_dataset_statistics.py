import copy

from vla.datasets.rlds.dataset import _share_bridge_action_statistics


def test_share_bridge_action_statistics_uses_an_independent_copy() -> None:
    bridge_action = {"q01": [-1.0], "q99": [1.0], "mask": [True]}
    statistics = {
        "bridge_orig": {"action": bridge_action},
        "widowx_simpler_rgbd": {"action": {"q01": [-2.0], "q99": [2.0]}},
    }

    _share_bridge_action_statistics(statistics)

    assert statistics["widowx_simpler_rgbd"]["action"] == bridge_action
    assert statistics["widowx_simpler_rgbd"]["action"] is not bridge_action

    statistics["widowx_simpler_rgbd"]["action"]["q01"].append(0.0)
    assert bridge_action["q01"] == [-1.0]


def test_share_bridge_action_statistics_ignores_other_mixtures() -> None:
    statistics = {"bridge_orig": {"action": {"q01": [-1.0]}}}
    original = copy.deepcopy(statistics)

    _share_bridge_action_statistics(statistics)

    assert statistics == original
