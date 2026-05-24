from typing import Tuple, Union

import torch

from .corrector import Corrector
from .decoder import Decoder, FullyConvolutionalDecoder
from .encoder import Encoder, FullyConvolutionalEncoder
from .initializer import Learned, SlotInitializer
from .model import SAVi
from .predictor import Predictor, TransformerPredictor


def build_savi(cfg, image_size: Union[int, Tuple[int, int]]) -> SAVi:
    if isinstance(image_size, int):
        image_size = (image_size, image_size)

    corrector_cfg = cfg.corrector
    encoder_cfg = cfg.encoder
    decoder_cfg = cfg.decoder
    predictor_cfg = cfg.predictor

    kernel_size = int(encoder_cfg.kernel_sizes[0])
    stride = int(encoder_cfg.strides[0])
    decoder_kernel_size = int(decoder_cfg.kernel_sizes[0])

    corrector = Corrector(
        num_slots=corrector_cfg.num_slots,
        slot_dim=corrector_cfg.slot_dim,
        feature_dim=corrector_cfg.feature_dim,
        num_iterations=corrector_cfg.num_iterations,
        num_initial_iterations=corrector_cfg.num_initial_iterations,
        hidden_dim=corrector_cfg.hidden_dim,
    )
    predictor = TransformerPredictor(
        slot_dim=corrector_cfg.slot_dim,
        action_dim=-1,
        num_heads=predictor_cfg.num_heads,
        mlp_size=predictor_cfg.mlp_size,
    )
    encoder = FullyConvolutionalEncoder(
        image_size=image_size,
        num_channels=list(encoder_cfg.num_channels),
        kernel_size=kernel_size,
        feature_dim=encoder_cfg.feature_dim,
        stride=stride,
        batch_norm=False,
        max_pool=False,
    )
    decoder = FullyConvolutionalDecoder(
        image_size=image_size,
        slot_dim=corrector_cfg.slot_dim,
        in_channels=corrector_cfg.slot_dim,
        num_channels=list(decoder_cfg.num_channels),
        kernel_size=decoder_kernel_size,
        stride=stride,
    )
    slot_initializer = Learned(
        num_slots=corrector_cfg.num_slots,
        slot_dim=corrector_cfg.slot_dim,
    )
    return SAVi(corrector, predictor, encoder, decoder, slot_initializer)


def load_savi_from_ckpt(cfg, ckpt_path: str, image_size: Union[int, Tuple[int, int]], device: str = "cpu") -> SAVi:
    model = build_savi(cfg=cfg, image_size=image_size)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model_state = {}
    for key, value in state_dict.items():
        if key.startswith("savi."):
            model_state[key[len("savi."):]] = value
        else:
            model_state[key] = value
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    return model
