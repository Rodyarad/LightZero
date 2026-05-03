from types import SimpleNamespace

import gym


class EnsureSpecIdWrapper(gym.Wrapper):
    """Provide gym-compatible ``env.spec.id`` for wrappers like ``RecordVideo``."""

    def __init__(self, env: gym.Env, fallback_id: str):
        super().__init__(env)
        # gym.wrappers.RecordVideo expects both spec.id and spec.kwargs.
        self._fallback_spec = SimpleNamespace(id=fallback_id, kwargs={})

    @property
    def spec(self):
        spec = getattr(self.env, "spec", None)
        if spec is not None and getattr(spec, "id", None) is not None:
            return spec
        return self._fallback_spec
