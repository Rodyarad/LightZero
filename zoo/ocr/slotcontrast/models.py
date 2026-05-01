from typing import Any, Dict, Optional

import torch
import torchvision
from torch import nn

from zoo.ocr.slotcontrast import configuration, modules, utils


def build_for_inference(model_config):
    
    initializer = modules.build_initializer(model_config.initializer)
    encoder = modules.build_encoder(model_config.encoder, "FrameEncoder")
    grouper = modules.build_grouper(model_config.grouper)
    decoder = modules.build_decoder(model_config.decoder)

    target_encoder = None
    if model_config.target_encoder:
        target_encoder = modules.build_encoder(model_config.target_encoder, "FrameEncoder")

    dynamics_predictor = None
    if model_config.dynamics_predictor:
        dynamics_predictor = modules.build_dynamics_predictor(model_config.dynamics_predictor)

    input_type = model_config.get("input_type", "image")
    if input_type == "image":
        processor = modules.LatentProcessor(grouper, predictor=None)
    elif input_type == "video":
        encoder = modules.MapOverTime(encoder)
        decoder = modules.MapOverTime(decoder)
        if target_encoder:
            target_encoder = modules.MapOverTime(target_encoder)
        if model_config.predictor is not None:
            predictor = modules.build_module(model_config.predictor)
        else:
            predictor = None
        if model_config.latent_processor:
            processor = modules.build_video(
                model_config.latent_processor,
                "LatentProcessor",
                corrector=grouper,
                predictor=predictor,
            )
        else:
            processor = modules.LatentProcessor(grouper, predictor)
        processor = modules.ScanOverTime(processor)
    else:
        raise ValueError(f"Unknown input type {input_type}")

    return SlotContrastModel(
        initializer=initializer,
        encoder=encoder,
        processor=processor,
        decoder=decoder,
        target_encoder=target_encoder,
        dynamics_predictor=dynamics_predictor,
        input_type=input_type,
    )


class SlotContrastModel(nn.Module):

    def __init__(
        self,
        initializer: nn.Module,
        encoder: nn.Module,
        processor: nn.Module,
        decoder: nn.Module,
        *,
        target_encoder: Optional[nn.Module] = None,
        dynamics_predictor: Optional[nn.Module] = None,
        input_type: str = "video",
    ):
        super().__init__()
        self.initializer = initializer
        self.encoder = encoder
        self.processor = processor
        self.decoder = decoder
        self.target_encoder = target_encoder
        self.dynamics_predictor = dynamics_predictor

        if input_type == "image":
            self.input_key = "image"
        elif input_type == "video":
            self.input_key = "video"
        else:
            raise ValueError(f"Unknown input type {input_type}")

        self._normalization = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        encoder_input = inputs[self.input_key]
        batch_size = len(encoder_input)

        encoder_output = self.encoder(encoder_input)
        features = encoder_output["features"]

        slots_initial = self.initializer(batch_size=batch_size)
        processor_output = self.processor(slots_initial, features)
        slots = processor_output["state"]
        decoder_output = self.decoder(slots)

        outputs = {
            "batch_size": batch_size,
            "encoder": encoder_output,
            "processor": processor_output,
            "decoder": decoder_output,
        }

        if self.dynamics_predictor:
            outputs["dynamics_predictor"] = self.dynamics_predictor(slots)

        return outputs

    @torch.no_grad()
    def extract_slots(self, image: torch.Tensor, prev_slots: Optional[torch.Tensor] = None) -> torch.Tensor:
        image = self._normalization(image)

        raw_encoder = self.encoder.module if isinstance(self.encoder, modules.MapOverTime) else self.encoder
        encoder_output = raw_encoder(image)
        features = encoder_output["features"]

        batch_size = image.shape[0]
        lp = self.processor.module if isinstance(self.processor, modules.ScanOverTime) else self.processor

        if prev_slots is None:
            slots = self.initializer(batch_size)
            is_first = True
        else:
            if lp.predictor is not None:
                slots = lp.predictor(prev_slots)
            else:
                slots = prev_slots
            is_first = False

        if is_first and lp.first_step_corrector_args:
            corrector_output = lp.corrector(slots, features, **lp.first_step_corrector_args)
        else:
            corrector_output = lp.corrector(slots, features)

        return corrector_output["slots"]


def load_from_checkpoint(config_path: str, checkpoint_path: str, device: str = "cpu") -> SlotContrastModel:

    config = configuration.load_config(config_path)
    model = build_for_inference(config.model)

    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    state_dict = checkpoint.get("state_dict", checkpoint)

    model_keys = set(model.state_dict().keys())
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}

    model.load_state_dict(filtered_state_dict)
    model.eval()
    model.to(device)

    return model
