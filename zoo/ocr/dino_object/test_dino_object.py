import numpy as np
import torch

from zoo.ocr.dino_object.encoder import DinoObjectEncoder
from zoo.ocr.dino_object.segmentation import maniskill_own_seg_v1


class FakeBackbone(torch.nn.Module):
    num_features = 384

    def forward_features(self, images):
        # Make it easy to check that each full-canvas masked image is separate.
        features = images.mean(dim=(2, 3)).mean(dim=1, keepdim=True)
        return features[:, None, :].expand(-1, 1, 384)


def test_masked_images_in_fixed_order(monkeypatch):
    monkeypatch.setattr("timm.create_model", lambda *args, **kwargs: FakeBackbone())
    encoder = DinoObjectEncoder(num_objects=2)
    image = np.full((28, 28, 3), 255, dtype=np.uint8)
    labels = np.ones((28, 28), dtype=np.uint8)
    labels[:, :7] = 2
    embeddings = encoder(image, labels)
    assert embeddings.shape == (2, 384)
    assert embeddings.dtype == np.float32
    assert embeddings[0, 0] > embeddings[1, 0]


def test_missing_group_gets_zero_embedding(monkeypatch):
    monkeypatch.setattr("timm.create_model", lambda *args, **kwargs: FakeBackbone())
    encoder = DinoObjectEncoder(num_objects=2)
    embeddings = encoder(np.zeros((28, 28, 3), dtype=np.uint8), np.ones((28, 28), dtype=np.uint8))
    assert np.any(embeddings[0])
    assert not np.any(embeddings[1])


def test_maniskill_target_has_own_group():
    class Object:
        def __init__(self, name):
            self.name = name

    ids = {1: Object("ground"), 2: Object("panda_link0"), 3: Object("panda_hand"),
           4: Object("table-workspace"), 5: Object("cube"), 6: Object("goal_region")}
    raw = np.arange(1, 7, dtype=np.int16)[None, :]
    np.testing.assert_array_equal(maniskill_own_seg_v1(raw, ids), [[2, 2, 1, 3, 4, 5]])
