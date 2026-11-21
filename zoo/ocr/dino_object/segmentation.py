"""Deterministic own_seg_v1 labels; 0 is reserved for unmapped pixels."""

import numpy as np


ROBOSUITE_NAMES = ("Panda + gripper", "Walls + floor + mount", "Table", "Cube")
MANISKILL_NAMES = ("Panda + gripper", "Ground + static base", "Table", "Cube", "Target zone")


def robosuite_own_seg_v1(element_labels, env):
    grouped = np.zeros(element_labels.shape, dtype=np.uint8)
    for geom_id in np.unique(element_labels):
        if geom_id < 0:
            continue
        geom_id = int(geom_id)
        name = env.sim.model.geom_id2name(geom_id)
        instance = env.model.geom_ids_to_instances.get(geom_id)
        if instance in ("Panda0", "PandaGripper0"):
            group = 1
        elif instance == "RethinkMount0" or name == "floor" or name.startswith("wall_"):
            group = 2
        elif name.startswith("table_"):
            group = 3
        elif instance == "cube":
            group = 4
        else:
            raise ValueError(f"Unclassified robosuite geom {geom_id}: {name!r}")
        grouped[element_labels == geom_id] = group
    if np.any(grouped == 0):
        raise ValueError("own_seg_v1 does not cover all robosuite pixels")
    return grouped


def maniskill_own_seg_v1(raw_labels, id_map):
    grouped = np.zeros(raw_labels.shape, dtype=np.uint8)
    for raw_id in np.unique(raw_labels):
        raw_id = int(raw_id)
        obj = id_map.get(raw_id)
        if obj is None:
            raise ValueError(f"Unknown ManiSkill segmentation ID: {raw_id}")
        name = obj.name
        if name in ("panda_link0", "ground"):
            group = 2
        elif name.startswith("panda_"):
            group = 1
        elif name == "table-workspace":
            group = 3
        elif name == "cube":
            group = 4
        elif name == "goal_region":
            group = 5
        else:
            raise ValueError(f"Unclassified ManiSkill actor/link: {raw_id} ({name})")
        grouped[raw_labels == raw_id] = group
    if np.any(grouped == 0):
        raise ValueError("own_seg_v1 does not cover all ManiSkill pixels")
    return grouped
