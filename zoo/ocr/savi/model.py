import math
from typing import Dict, Optional

import torch
import torch.nn as nn

from .blocks import init_xavier_
from .corrector import Corrector
from .decoder import Decoder
from .encoder import Encoder
from .initializer import SlotInitializer
from .predictor import Predictor


class SAVi(nn.Module):
    def __init__(
        self,
        corrector: Corrector,
        predictor: Predictor,
        encoder: Encoder,
        decoder: Decoder,
        initializer: SlotInitializer,
    ) -> None:
        super().__init__()
        self.corrector = corrector
        self.predictor = predictor
        self.encoder = encoder
        self.decoder = decoder
        self.initializer = initializer
        self.num_slots = corrector.num_slots
        self.slot_dim = corrector.slot_dim
        self._initialize_parameters()

    @torch.no_grad()
    def _initialize_parameters(self):
        init_xavier_(self)
        torch.nn.init.zeros_(self.corrector.gru.bias_ih)
        torch.nn.init.zeros_(self.corrector.gru.bias_hh)
        torch.nn.init.orthogonal_(self.corrector.gru.weight_hh)
        if hasattr(self.corrector, "slots_mu"):
            limit = math.sqrt(6.0 / (1 + self.corrector.slot_dim))
            torch.nn.init.uniform_(self.corrector.slots_mu, -limit, limit)
            torch.nn.init.uniform_(self.corrector.slots_sigma, -limit, limit)

    @staticmethod
    def _empty_actions_like(slots: torch.Tensor) -> torch.Tensor:
        return torch.empty(slots.shape[0], 0, device=slots.device, dtype=slots.dtype)

    @torch.no_grad()
    def extract_slots(self, images: torch.Tensor, prev_slots: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = images.shape[0]
        if prev_slots is None:
            predicted_slots = self.initializer(batch_size=batch_size)
            step = 0
        else:
            predicted_slots = self.predictor(prev_slots, self._empty_actions_like(prev_slots))
            step = 1

        features = self.encoder(images)
        return self.corrector(features, slots=predicted_slots, step=step)

    def decode(self, slots: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, sequence_length, num_slots, slot_dim = slots.shape
        if num_slots != self.num_slots or slot_dim != self.slot_dim:
            raise ValueError(
                f"Unexpected slot shape {slots.shape}. Expected (*, {self.num_slots}, {self.slot_dim})."
            )

        rgbs, masks = self.decoder(slots.flatten(end_dim=1))
        rgbs = rgbs.view(batch_size, sequence_length, num_slots, 3, *self.decoder.image_size)
        masks = masks.view(batch_size, sequence_length, num_slots, 1, *self.decoder.image_size)
        return {
            "reconstructions": torch.sum(rgbs * masks, dim=2).clamp(0, 1),
            "rgbs": rgbs,
            "masks": masks,
        }

    def forward(
        self,
        images: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        prior_slots: Optional[torch.Tensor] = None,
        step_offset: int = 0,
        reconstruct: bool = True,
        **kwargs,
    ):
        slots_sequence = []
        reconstruction_sequence = []
        rgbs_sequence = []
        masks_sequence = []

        sequence_length = images.shape[1]
        if actions is None:
            actions = torch.empty(
                images.shape[0],
                max(sequence_length - 1, 0),
                0,
                device=images.device,
                dtype=images.dtype,
            )

        if prior_slots is None:
            predicted_slots = self.initializer(batch_size=images.shape[0])
        else:
            first_action = actions[:, 0] if sequence_length > 1 else self._empty_actions_like(prior_slots)
            predicted_slots = self.predictor(prior_slots, first_action)

        for t in range(sequence_length):
            img_feats = self.encoder(images[:, t])
            slots = self.corrector(img_feats, slots=predicted_slots, step=t + step_offset)
            if t < sequence_length - 1:
                predicted_slots = self.predictor(slots, actions[:, t])
            slots_sequence.append(slots)
            if reconstruct:
                rgb, masks = self.decoder(slots)
                reconstruction_sequence.append(torch.sum(rgb * masks, dim=1))
                rgbs_sequence.append(rgb)
                masks_sequence.append(masks)

        slots_sequence = torch.stack(slots_sequence, dim=1)
        if not reconstruct:
            return slots_sequence

        return (
            slots_sequence,
            torch.stack(reconstruction_sequence, dim=1),
            torch.stack(rgbs_sequence, dim=1),
            torch.stack(masks_sequence, dim=1),
        )
