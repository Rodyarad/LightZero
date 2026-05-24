from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import nn

from .dvae import dVAE
from .transformer import TransformerDecoder, TransformerEncoder
from .utils import Conv2dBlock, conv2d, gru_cell, gumbel_softmax, linear


class SlotAttentionVideo(nn.Module):
    def __init__(
        self,
        num_iterations: int,
        num_slots: int,
        input_size: int,
        slot_size: int,
        mlp_hidden_size: int,
        num_predictor_blocks: int = 1,
        num_predictor_heads: int = 4,
        dropout: float = 0.1,
        epsilon: float = 1e-8,
    ):
        super().__init__()
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.epsilon = epsilon

        self.slot_mu = nn.Parameter(torch.Tensor(1, 1, slot_size))
        self.slot_log_sigma = nn.Parameter(torch.Tensor(1, 1, slot_size))
        nn.init.xavier_uniform_(self.slot_mu)
        nn.init.xavier_uniform_(self.slot_log_sigma)

        self.norm_inputs = nn.LayerNorm(input_size)
        self.norm_slots = nn.LayerNorm(slot_size)
        self.norm_mlp = nn.LayerNorm(slot_size)

        self.project_q = linear(slot_size, slot_size, bias=False)
        self.project_k = linear(input_size, slot_size, bias=False)
        self.project_v = linear(input_size, slot_size, bias=False)

        self.gru = gru_cell(slot_size, slot_size)
        self.mlp = nn.Sequential(
            linear(slot_size, mlp_hidden_size, weight_init="kaiming"),
            nn.ReLU(),
            linear(mlp_hidden_size, slot_size),
        )
        self.predictor = TransformerEncoder(
            num_predictor_blocks, slot_size, num_predictor_heads, dropout
        )

    def _init_slots(self, batch_size: int, reference: torch.Tensor) -> torch.Tensor:
        slots = reference.new_empty(batch_size, self.num_slots, self.slot_size).normal_()
        return self.slot_mu + torch.exp(self.slot_log_sigma) * slots

    def step(
        self,
        inputs_t: torch.Tensor,
        prev_slots: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs_t = self.norm_inputs(inputs_t)
        k = self.project_k(inputs_t) * (self.slot_size ** (-0.5))
        v = self.project_v(inputs_t)

        if prev_slots is None:
            slots = self._init_slots(inputs_t.shape[0], inputs_t)
        else:
            slots = self.predictor(prev_slots)

        for i in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)
            q = self.project_q(slots)
            attn_logits = torch.bmm(k, q.transpose(-1, -2))
            attn_vis = F.softmax(attn_logits, dim=-1)

            attn = attn_vis + self.epsilon
            attn = attn / torch.sum(attn, dim=-2, keepdim=True)
            updates = torch.bmm(attn.transpose(-1, -2), v)

            slots = self.gru(updates.reshape(-1, self.slot_size), slots_prev.reshape(-1, self.slot_size))
            slots = slots.reshape(-1, self.num_slots, self.slot_size)

            if i < self.num_iterations - 1:
                slots = slots + self.mlp(self.norm_mlp(slots))

        return slots, attn_vis

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, timesteps, _, _ = inputs.size()
        slots_prev = None
        slots_collect, attns_collect = [], []
        for t in range(timesteps):
            slots, attn_vis = self.step(inputs[:, t], prev_slots=slots_prev)
            slots_collect.append(slots)
            attns_collect.append(attn_vis)
            slots_prev = slots
        return torch.stack(slots_collect, dim=1), torch.stack(attns_collect, dim=1)


class LearnedPositionalEmbedding1D(nn.Module):
    def __init__(self, num_inputs: int, input_size: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Parameter(torch.zeros(1, num_inputs, input_size), requires_grad=True)
        nn.init.trunc_normal_(self.pe)

    def forward(self, input_tensor: torch.Tensor, offset: int = 0) -> torch.Tensor:
        timesteps = input_tensor.shape[1]
        return self.dropout(input_tensor + self.pe[:, offset:offset + timesteps])


class CartesianPositionalEmbedding(nn.Module):
    def __init__(self, channels: int, image_size: int):
        super().__init__()
        self.projection = conv2d(4, channels, 1)
        self.pe = nn.Parameter(self._build_grid(image_size).unsqueeze(0), requires_grad=False)

    @staticmethod
    def _build_grid(side_length: int) -> torch.Tensor:
        coords = torch.linspace(0.0, 1.0, side_length + 1)
        coords = 0.5 * (coords[:-1] + coords[1:])
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
        return torch.stack((grid_x, grid_y, 1 - grid_x, 1 - grid_y), dim=0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.projection(self.pe)


class OneHotDictionary(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int):
        super().__init__()
        self.dictionary = nn.Embedding(vocab_size, emb_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dictionary(torch.argmax(x, dim=-1))


class STEVEEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        stride = 1 if args.image_size == 64 else 2
        self.cnn = nn.Sequential(
            Conv2dBlock(args.img_channels, args.cnn_hidden_size, 5, stride, 2),
            Conv2dBlock(args.cnn_hidden_size, args.cnn_hidden_size, 5, 1, 2),
            Conv2dBlock(args.cnn_hidden_size, args.cnn_hidden_size, 5, 1, 2),
            conv2d(args.cnn_hidden_size, args.d_model, 5, 1, 2),
        )
        pos_size = args.image_size if args.image_size == 64 else args.image_size // 2
        self.pos = CartesianPositionalEmbedding(args.d_model, pos_size)
        self.layer_norm = nn.LayerNorm(args.d_model)
        self.mlp = nn.Sequential(
            linear(args.d_model, args.d_model, weight_init="kaiming"),
            nn.ReLU(),
            linear(args.d_model, args.d_model),
        )
        self.savi = SlotAttentionVideo(
            args.num_iterations,
            args.num_slots,
            args.d_model,
            args.slot_size,
            args.mlp_hidden_size,
            args.num_predictor_blocks,
            args.num_predictor_heads,
            args.predictor_dropout,
        )
        self.slot_proj = linear(args.slot_size, args.d_model, bias=False)


class STEVEDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dict = OneHotDictionary(args.vocab_size, args.d_model)
        self.bos = nn.Parameter(torch.Tensor(1, 1, args.d_model))
        nn.init.xavier_uniform_(self.bos)
        self.pos = LearnedPositionalEmbedding1D(1 + (args.image_size // 4) ** 2, args.d_model)
        self.tf = TransformerDecoder(
            args.num_decoder_blocks,
            (args.image_size // 4) ** 2,
            args.d_model,
            args.num_decoder_heads,
            args.dropout,
        )
        self.head = linear(args.d_model, args.vocab_size, bias=False)


class STEVE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_slots = args.num_slots
        self.slot_size = args.slot_size
        self.image_size = args.image_size
        self.vocab_size = args.vocab_size
        self.d_model = args.d_model

        self.dvae = dVAE(args.vocab_size, args.img_channels)
        self.steve_encoder = STEVEEncoder(args)
        self.steve_decoder = STEVEDecoder(args)

    def _encode_embeddings(self, video: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        batch_size, timesteps, _, _, _ = video.size()
        video_flat = video.flatten(end_dim=1)
        emb = self.steve_encoder.pos(self.steve_encoder.cnn(video_flat))
        h_enc, w_enc = emb.shape[-2:]
        emb_set = emb.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)
        emb_set = self.steve_encoder.mlp(self.steve_encoder.layer_norm(emb_set))
        emb_set = emb_set.reshape(batch_size, timesteps, h_enc * w_enc, self.d_model)
        return emb_set, h_enc, w_enc

    def _attn_to_vis(self, video: torch.Tensor, attns: torch.Tensor, h_enc: int, w_enc: int) -> torch.Tensor:
        _, _, _, h, w = video.shape
        attns = (
            attns.transpose(-1, -2)
            .reshape(video.shape[0], video.shape[1], self.num_slots, 1, h_enc, w_enc)
            .repeat_interleave(h // h_enc, dim=-2)
            .repeat_interleave(w // w_enc, dim=-1)
        )
        return video.unsqueeze(2) * attns + (1.0 - attns)

    def forward(self, video: torch.Tensor, tau: float, hard: bool):
        batch_size, timesteps, channels, h, w = video.size()
        video_flat = video.flatten(end_dim=1)

        z_logits = F.log_softmax(self.dvae.encoder(video_flat), dim=1)
        z_soft = gumbel_softmax(z_logits, tau, hard, dim=1)
        z_hard = gumbel_softmax(z_logits, tau, True, dim=1).detach()
        z_hard = z_hard.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)

        z_emb = self.steve_decoder.dict(z_hard)
        z_emb = torch.cat([self.steve_decoder.bos.expand(batch_size * timesteps, -1, -1), z_emb], dim=1)
        z_emb = self.steve_decoder.pos(z_emb)

        dvae_recon = self.dvae.decoder(z_soft).reshape(batch_size, timesteps, channels, h, w)
        dvae_mse = ((video - dvae_recon) ** 2).sum() / (batch_size * timesteps)

        emb_set, h_enc, w_enc = self._encode_embeddings(video)
        slots, attns = self.steve_encoder.savi(emb_set)
        attns_vis = self._attn_to_vis(video, attns, h_enc, w_enc)

        projected_slots = self.steve_encoder.slot_proj(slots)
        pred = self.steve_decoder.tf(z_emb[:, :-1], projected_slots.flatten(end_dim=1))
        pred = self.steve_decoder.head(pred)
        cross_entropy = -(z_hard * torch.log_softmax(pred, dim=-1)).sum() / (batch_size * timesteps)

        return dvae_recon.clamp(0.0, 1.0), cross_entropy, dvae_mse, attns_vis

    def encode(
        self,
        video: torch.Tensor,
        initial_slots: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        emb_set, h_enc, w_enc = self._encode_embeddings(video)
        slots_prev = initial_slots
        slots_collect, attn_collect = [], []
        for t in range(video.shape[1]):
            slots, attn_vis = self.steve_encoder.savi.step(emb_set[:, t], prev_slots=slots_prev)
            slots_collect.append(slots)
            attn_collect.append(attn_vis)
            slots_prev = slots
        slots_seq = torch.stack(slots_collect, dim=1)
        attn_raw = torch.stack(attn_collect, dim=1)
        attn_vis = self._attn_to_vis(video, attn_raw, h_enc, w_enc)
        return slots_seq, attn_vis, attn_raw

    def decode(self, slots: torch.Tensor) -> torch.Tensor:
        batch_size = slots.size(0)
        h_enc = w_enc = self.image_size // 4
        gen_len = h_enc * w_enc

        slots = self.steve_encoder.slot_proj(slots)
        z_gen = slots.new_zeros((batch_size, 0, self.vocab_size))
        decoder_input = self.steve_decoder.bos.expand(batch_size, 1, -1)
        for _ in range(gen_len):
            decoder_output = self.steve_decoder.tf(self.steve_decoder.pos(decoder_input), slots)
            z_next = F.one_hot(
                self.steve_decoder.head(decoder_output)[:, -1:].argmax(dim=-1),
                self.vocab_size,
            )
            z_gen = torch.cat((z_gen, z_next), dim=1)
            decoder_input = torch.cat((decoder_input, self.steve_decoder.dict(z_next)), dim=1)

        z_gen = z_gen.transpose(1, 2).float().reshape(batch_size, -1, h_enc, w_enc)
        return self.dvae.decoder(z_gen).clamp(0.0, 1.0)

    def reconstruct_autoregressive(self, video: torch.Tensor) -> torch.Tensor:
        batch_size, timesteps, channels, h, w = video.size()
        slots, _, _ = self.encode(video)
        recon = self.decode(slots.flatten(end_dim=1))
        return recon.reshape(batch_size, timesteps, channels, h, w)

    @torch.no_grad()
    def extract_slots(
        self,
        images: torch.Tensor,
        prev_slots: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if images.dim() != 4:
            raise ValueError(f"Expected images shape [B,C,H,W], got {tuple(images.shape)}")
        emb = self.steve_encoder.pos(self.steve_encoder.cnn(images))
        emb = emb.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)
        emb = self.steve_encoder.mlp(self.steve_encoder.layer_norm(emb))
        slots, _ = self.steve_encoder.savi.step(emb, prev_slots=prev_slots)
        return slots

    @torch.no_grad()
    def get_samples(
        self,
        images: torch.Tensor,
        prev_slots: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if images.dim() != 4:
            raise ValueError(f"Expected images shape [B,C,H,W], got {tuple(images.shape)}")
        emb = self.steve_encoder.pos(self.steve_encoder.cnn(images))
        h_enc, w_enc = emb.shape[-2:]
        emb = emb.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)
        emb = self.steve_encoder.mlp(self.steve_encoder.layer_norm(emb))
        slots, attn_raw = self.steve_encoder.savi.step(emb, prev_slots=prev_slots)
        attn_vis = self._attn_to_vis(images.unsqueeze(1), attn_raw.unsqueeze(1), h_enc, w_enc)[:, 0]
        preview = torch.cat([images] + [attn_vis[:, i] for i in range(attn_vis.shape[1])], dim=-1)
        preview = (preview.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(np.uint8)
        return {"slots": slots, "samples": preview}


def _cfg_to_namespace(cfg) -> SimpleNamespace:
    data = OmegaConf.to_container(cfg, resolve=True)
    return SimpleNamespace(**data)


def load_steve_from_ckpt(
    config_path: str,
    checkpoint_path: str,
    image_size: Union[int, Tuple[int, int]],
    image_channels: int = 3,
    device: str = "cpu",
) -> STEVE:
    cfg = OmegaConf.load(config_path)
    if isinstance(image_size, tuple):
        if image_size[0] != image_size[1]:
            raise ValueError(f"STEVE expects square frames, got image_size={image_size}")
        cfg.image_size = int(image_size[0])
    else:
        cfg.image_size = int(image_size)
    cfg.img_channels = int(image_channels)

    model = STEVE(_cfg_to_namespace(cfg))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    remapped = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    model.load_state_dict(remapped, strict=True)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def visualize_steve_samples(
    model: STEVE,
    images: torch.Tensor,
    prev_slots: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    return model.get_samples(images=images, prev_slots=prev_slots)
