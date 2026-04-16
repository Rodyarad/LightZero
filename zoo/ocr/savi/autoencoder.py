import math
import sys
import types
from zoo.ocr.savi.base import Autoencoder
import zoo.ocr.savi as _savi_pkg
import zoo.ocr.savi.cnn as _savi_cnn_pkg
from zoo.ocr.savi import Corrector, SlotInitializer, Predictor, SaviCnnEncoder, SaviCnnDecoder
from zoo.ocr.savi.predictor import init_xavier_, TransformerPredictor
from zoo.ocr.savi.initializer import Learned
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Union
from omegaconf import OmegaConf, DictConfig
from zoo.ocr.savi.visualizations import make_grid, make_row, stack_rows, create_segmentation_overlay, slot_color, draw_ticks


class SAVi(Autoencoder):
    def __init__(self, corrector: Corrector, predictor: Predictor, encoder: SaviCnnEncoder, decoder: SaviCnnDecoder,
                 initializer: SlotInitializer) -> None:
        super().__init__()
        self.corrector = corrector
        self.predictor = predictor
        self.encoder = encoder
        self.decoder = decoder
        self.initializer = initializer
        self._initialize_parameters()

    @torch.no_grad()
    def _initialize_parameters(self):
        init_xavier_(self)
        torch.nn.init.zeros_(self.corrector.gru.bias_ih)
        torch.nn.init.zeros_(self.corrector.gru.bias_hh)
        torch.nn.init.orthogonal_(self.corrector.gru.weight_hh)
        if hasattr(self.corrector, "slots_mu"):
            limit = math.sqrt(6.0 / (1 + self.corrector.dim_slots))
            torch.nn.init.uniform_(self.corrector.slots_mu, -limit, limit)
            torch.nn.init.uniform_(self.corrector.slots_sigma, -limit, limit)

    @property
    def num_slots(self) -> int:
        return self.corrector.num_slots

    @property
    def slot_dim(self) -> int:
        return self.corrector.slot_dim

    def encode(self, images: torch.Tensor, prior_slots: Optional[torch.Tensor] = None) -> torch.Tensor:
        slots_sequence = []
        batch_size, sequence_length, _, _, _ = images.size()

        predicted_slots = self.initializer(batch_size) if prior_slots is None else self.predictor(prior_slots)
        for time_step in range(sequence_length):
            features = self.encoder(images[:, time_step])
            slots = self.corrector(features, slots=predicted_slots, is_first=prior_slots is None and time_step == 0)
            if time_step < sequence_length - 1:
                predicted_slots = self.predictor(slots)
            slots_sequence.append(slots)

        slots_sequence = torch.stack(slots_sequence, dim=1)
        return slots_sequence

    @torch.no_grad()
    def extract_slots(self, images: torch.Tensor, prev_slots: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Extract slots from a single image (not a sequence).

        Args:
            images: (B, C, H, W) tensor.
            prev_slots: (B, num_slots, slot_dim) tensor or None.

        Returns:
            slots: (B, num_slots, slot_dim) tensor.
        """
        batch_size = images.shape[0]
        if prev_slots is None:
            predicted_slots = self.initializer(batch_size)
            is_first = True
        else:
            predicted_slots = self.predictor(prev_slots)
            is_first = False
        features = self.encoder(images)
        slots = self.corrector(features, slots=predicted_slots, is_first=is_first)
        return slots

    def decode(self, slots: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, sequence_length, num_slots, slot_dim = slots.size()
        rgbs, masks = self.decoder(slots.flatten(end_dim=1))
        rgbs = rgbs.view(batch_size, sequence_length, num_slots, 3, *self.decoder.image_size)
        masks = masks.view(batch_size, sequence_length, num_slots, 1, *self.decoder.image_size)
        return {"reconstructions": torch.sum(rgbs * masks, dim=2).clamp(0, 1), "rgbs": rgbs, "masks": masks}

    def forward(self, images: torch.Tensor, prior_slots: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        batch_size, sequence_length, num_channels, height, width = images.size()
        outputs = super().forward(images, prior_slots)
        outputs["rgbs"] = outputs["rgbs"].reshape(batch_size, sequence_length, self.num_slots, num_channels, height, width)
        outputs["masks"] = outputs["masks"].reshape(batch_size, sequence_length, self.num_slots, 1, height, width)
        return outputs

    @torch.no_grad()
    def visualize_reconstruction(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        sequence_length, num_slots, _, height, width = outputs["rgbs"].size()
        rows = []
        if "images" in outputs:
            rows.append(make_row(outputs["images"].cpu()))
        rows.append(make_row(outputs["reconstructions"].cpu()))
        images = outputs["images"].cpu() if "images" in outputs else torch.zeros_like(outputs["reconstructions"].cpu())
        rows.append(make_row(create_segmentation_overlay(images, outputs["masks"].cpu(),
                                                         background_brightness=0.0)))
        individual_slots = outputs["masks"].cpu() * outputs["rgbs"].cpu()
        rows.extend(make_row(individual_slots[:, slot_index], pad_color=slot_color(slot_index, num_slots))
                    for slot_index in range(num_slots))

        if "xticks" in outputs:
            label_backgrounds = torch.ones(sequence_length, 3, height // 3, width)
            labels = draw_ticks(label_backgrounds, outputs["xticks"], color=(0, 0, 0))
            rows.append(make_row(labels, pad_color=torch.tensor([1, 1, 1])))

        return stack_rows(rows)


def build_savi(cfg: DictConfig, image_size: Union[int, Tuple[int, int]]) -> SAVi:
    if isinstance(image_size, int):
        image_size = (image_size, image_size)

    c = cfg.corrector
    return SAVi(
        corrector=Corrector(
            num_slots=c.num_slots,
            slot_dim=c.slot_dim,
            feature_dim=c.feature_dim,
            hidden_dim=c.hidden_dim,
            num_iterations=c.num_iterations,
            num_initial_iterations=c.num_initial_iterations,
        ),
        predictor=TransformerPredictor(
            slot_dim=c.slot_dim,
            num_heads=cfg.predictor.num_heads,
            mlp_size=cfg.predictor.mlp_size,
        ),
        encoder=SaviCnnEncoder(
            image_size=image_size,
            num_channels=list(cfg.encoder.num_channels),
            kernel_sizes=list(cfg.encoder.kernel_sizes),
            strides=list(cfg.encoder.strides),
            feature_dim=cfg.encoder.feature_dim,
        ),
        decoder=SaviCnnDecoder(
            image_size=image_size,
            num_channels=list(cfg.decoder.num_channels),
            kernel_sizes=list(cfg.decoder.kernel_sizes),
            strides=list(cfg.decoder.strides),
            in_channels=c.slot_dim,
        ),
        initializer=Learned(
            num_slots=c.num_slots,
            slot_dim=c.slot_dim,
        ),
    )


def _remap_ckpt_keys(state_dict: dict) -> dict:
    """Remap legacy SAVi checkpoint keys to the current module layout."""
    mapping = {
        "encoder.conv.0.block.0": "encoder.encoder.0",
        "encoder.conv.1.block.0": "encoder.encoder.2",
        "encoder.conv.2.block.0": "encoder.encoder.4",
        "encoder.conv.3.block.0": "encoder.encoder.6",
        "encoder.positional_encoding.projection": "encoder.positional_embedding.projection",
        "encoder.mlp.0": "encoder.shared_mlp.0",
        "encoder.mlp.1": "encoder.shared_mlp.1",
        "encoder.mlp.3": "encoder.shared_mlp.3",
        "decoder.decoder.0.block.0": "decoder.encoder.0",
        "decoder.decoder.1.block.0": "decoder.encoder.2",
        "decoder.decoder.2.block.0": "decoder.encoder.4",
        "decoder.decoder.3.block.0": "decoder.encoder.6",
        "decoder.decoder.4": "decoder.encoder.8",
        "decoder.positional_encoding.projection": "decoder.positional_embedding.projection",
    }

    remapped = {}
    for old_key, value in state_dict.items():
        key = old_key
        if key.startswith("savi."):
            key = key[5:]
        if key.startswith("autoencoder."):
            key = key[12:]

        new_key = key
        for old_prefix, new_prefix in mapping.items():
            if key.startswith(old_prefix):
                new_key = key.replace(old_prefix, new_prefix, 1)
                break

        remapped[new_key] = value

    return remapped


def _register_legacy_module_aliases():
    """Map old 'modeling.autoencoder.savi.*' paths to 'zoo.ocr.savi.*'
    so that torch.load can unpickle Lightning checkpoints from the old repo."""
    _stub = types.ModuleType
    for name in ("modeling", "modeling.autoencoder"):
        if name not in sys.modules:
            sys.modules[name] = _stub(name)

    from zoo.ocr.savi import corrector, encoder, decoder, predictor, initializer
    from zoo.ocr.savi.cnn import encoder as cnn_encoder, decoder as cnn_decoder

    sys.modules["modeling.autoencoder.savi"] = _savi_pkg
    sys.modules["modeling.autoencoder.savi.corrector"] = corrector
    sys.modules["modeling.autoencoder.savi.encoder"] = encoder
    sys.modules["modeling.autoencoder.savi.decoder"] = decoder
    sys.modules["modeling.autoencoder.savi.predictor"] = predictor
    sys.modules["modeling.autoencoder.savi.initializer"] = initializer
    sys.modules["modeling.autoencoder.savi.autoencoder"] = sys.modules[__name__]
    sys.modules["modeling.autoencoder.savi.base"] = sys.modules["zoo.ocr.savi.base"]
    sys.modules["modeling.autoencoder.savi.cnn"] = _savi_cnn_pkg
    sys.modules["modeling.autoencoder.savi.cnn.encoder"] = cnn_encoder
    sys.modules["modeling.autoencoder.savi.cnn.decoder"] = cnn_decoder


def load_savi_from_ckpt(cfg: DictConfig,
                        ckpt_path: str,
                        image_size: Union[int, Tuple[int, int]],
                        device: str = "cpu") -> SAVi:
    model = build_savi(cfg, image_size=image_size)

    _register_legacy_module_aliases()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    state_dict = _remap_ckpt_keys(state_dict)
    model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()
    return model
