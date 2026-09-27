from typing import Any, List, Sequence, Tuple

import numpy as np
import torch


class SearchPathTracker:
    """
    Overview:
        Reconstructs, for every batch element of a batched MCTS, the root -> leaf chain of
        ``(latent_state, action)`` pairs of the node that is being expanded in the current simulation.

        In a batched MCTS the node expanded at simulation ``i`` is stored at index ``i + 1`` of
        ``latent_state_batch_in_search_path`` (index 0 holds the roots). Its parent for batch element
        ``iy`` is ``latent_state_index_in_search_path[iy]`` and it is reached through ``last_actions[iy]``.
        Remembering these two arrays per node is enough to walk back to the root.

        This is what a cache-free UniZero world model needs as temporal context for a node: the leaves
        of *previous simulations* (which the plain per-simulation history contains) belong to unrelated
        branches of the tree and must not be used as context.
    """

    def __init__(self, batch_size: int) -> None:
        self._batch_size = batch_size
        # node index -> np.ndarray (B,) of parent node indices; None for the root.
        self._parent_index: List[Any] = [None]
        # node index -> np.ndarray (B, ...) of actions leading to this node; None for the root.
        self._parent_action: List[Any] = [None]

    @staticmethod
    def _to_numpy(x: Any) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def build_paths(
            self,
            latent_state_batch_in_search_path: Sequence[Any],
            latent_state_index_in_search_path: Sequence[int],
            latent_state_index_in_batch: Sequence[int],
            last_actions: Any,
    ) -> List[Tuple[List[np.ndarray], List[np.ndarray]]]:
        """
        Overview:
            Return the root -> leaf ``(states, actions)`` chain for each batch element of the current
            simulation and register the node that this simulation is about to expand.
        Arguments:
            - latent_state_batch_in_search_path: list of per-node batched latent states, ``[node][iy]``.
            - latent_state_index_in_search_path: node index of the leaf for each traversed batch element.
            - latent_state_index_in_batch: batch index ``iy`` for each traversed batch element.
            - last_actions: actions selected at the leaves, ``(B,)`` for discrete or ``(B, A)`` for continuous.
        Returns:
            - paths: ``paths[iy] = (states, actions)`` with ``states[0]`` the root latent state and
              ``actions[-1]`` the action taken at the leaf. Both lists have length ``depth + 1``.
        """
        last_actions_np = self._to_numpy(last_actions)
        leaf_index = np.zeros(self._batch_size, dtype=np.int64)
        for ix, iy in zip(latent_state_index_in_search_path, latent_state_index_in_batch):
            leaf_index[iy] = ix

        paths: List[Tuple[List[np.ndarray], List[np.ndarray]]] = []
        for iy in range(self._batch_size):
            states: List[np.ndarray] = []
            actions: List[np.ndarray] = []
            node = int(leaf_index[iy])
            act = last_actions_np[iy]
            while True:
                states.append(np.asarray(latent_state_batch_in_search_path[node][iy]))
                actions.append(np.asarray(act))
                if node == 0:
                    break
                act = self._parent_action[node][iy]
                node = int(self._parent_index[node][iy])
            states.reverse()
            actions.reverse()
            paths.append((states, actions))

        # The node expanded in this simulation gets index len(self._parent_index).
        self._parent_index.append(leaf_index)
        self._parent_action.append(last_actions_np)
        return paths
