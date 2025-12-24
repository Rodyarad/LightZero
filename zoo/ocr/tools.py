import math
import os
from functools import partial
from pathlib import Path

# Types
from typing import Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import omegaconf
import torch
import torchvision
import logging
import omegaconf
import numpy as np
from pathlib import Path

import wandb
from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from zoo.ocr.dinosaur.conditioning import RandomConditioning
from zoo.ocr.dinosaur.decoding import PatchDecoder
from zoo.ocr.dinosaur.feature_extractors.timm import TimmFeatureExtractor
from zoo.ocr.dinosaur.neural_networks import build_two_layer_mlp, build_mlp
from zoo.ocr.dinosaur.neural_networks.positional_embedding import DummyPositionEmbed
from zoo.ocr.dinosaur.neural_networks.wrappers import Sequential
from zoo.ocr.dinosaur.perceptual_grouping import SlotAttentionGrouping

Tensor = TypeVar("torch.tensor")
NN = TypeVar("torch.nn")


# [B, D, H, W] -> [B, N, D]
img_to_slot = lambda x: x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, x.shape[1])


# [B, N, D] -> [B, D, H, W]
def slot_to_img(slot):
    B, N, D = slot.shape
    size = int(math.sqrt(N))
    return slot.reshape(B, size, size, D).permute(0, 3, 1, 2)


def get_ocr_checkpoint_path(config):
    if config.local_file:
        return Path(__file__).resolve().parents[1] / config.local_file

    entity = config.entity
    project = config.project
    run_id = config.run_id
    file_path = config.file
    ocr_dir = Path(wandb.run.dir) / "ocr_checkpoints"
    ocr_dir.mkdir(parents=True, exist_ok=True)
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    run.file(file_path).download(root=ocr_dir, replace=True)
    return ocr_dir / file_path


def get_log_prefix(config):
    prefix = ""
    if config.ocr.name == "VAE":
        if config.ocr.use_cnn_feat:
            prefix = f"{config.ocr.name}N{config.ocr.cnn_feat_size**2}"
        else:
            prefix = f"{config.ocr.name}"
    elif config.ocr.name == "SlotAttn" or config.ocr.name == "SLATE":
        prefix = f"{config.ocr.name}N{config.ocr.slotattr.num_slots}"
    elif config.ocr.name == "MoNet":
        prefix = f"{config.ocr.name}N{config.ocr.num_slots}"
    else:
        prefix = f"{config.ocr.name}"
    if hasattr(config, "pooling"):
        if config.pooling.ocr_checkpoint.run_id != "":
            prefix = "Pretrained-" + prefix
        if config.pooling.learn_aux_loss:
            prefix += f"Aux"
        if config.pooling.learn_downstream_loss:
            prefix += f"FineTune"
        prefix += f"-{config.pooling.name}"
    return prefix


def init_wandb(config, log_name, tags="", sync_tensorboard=None, monitor_gym=None):
    if config.wandb.offline:
        os.environ["WANDB_MODE"] = "offline"
    else:
        os.environ["WANDB_MODE"] = "online"
    wandb.config = omegaconf.OmegaConf.to_container(
        config, resolve=True, throw_on_missing=True
    )
    run = wandb.init(
        entity=config.wandb.entity,
        project=config.wandb.project,
        config=wandb.config,
        name=log_name,
        dir=str(Path.cwd().parent.parent),
        save_code=True,
        sync_tensorboard=sync_tensorboard,
        monitor_gym=monitor_gym,
        resume="allow",
        id=config.wandb.run_id,
        tags=tags,
    )
    model_dir = Path(run.dir) / "checkpoints"
    model_dir.mkdir(parents=True, exist_ok=True)


# https://discuss.pytorch.org/t/moving-optimizer-from-cpu-to-gpu/96068/3
def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


# Taken from https://github.com/lcswillems/torch-ac/blob/master/torch_ac/utils/dictlist.py
class DictList(dict):
    """A dictionnary of lists of same size. Dictionnary items can be
    accessed using `.` notation and list items using `[]` notation.
    Example:
        >>> d = DictList({"a": [[1, 2], [3, 4]], "b": [[5], [6]]})
        >>> d.a
        [[1, 2], [3, 4]]
        >>> d[0]
        DictList({"a": [1, 2], "b": [5]})
    """

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __len__(self):
        return len(next(iter(dict.values(self))))

    def __getitem__(self, index):
        return DictList({key: value[index] for key, value in dict.items(self)})

    def __setitem__(self, index, d):
        for key, value in d.items():
            dict.__getitem__(self, key)[index] = value


def preprocessing_obs(obs, device, type="image"):
    ret = torch.Tensor(obs.copy()).to(device).unsqueeze(0)
    if type == "image":
        return ret.permute(0, 3, 1, 2) / 255
    elif type == "state":
        return ret


# upload batch to working device
def to_device(batch, device):
    if type(batch) == type([]):
        for i in range(len(batch)):
            batch[i] = batch[i].to(device)
    elif type(batch) == type({}):
        for k in batch.keys():
            batch[k] = batch[k].to(device)
    else:
        batch = batch.to(device)
    return batch


# get_item from pytorch tensor
def get_item(x):
    if len(x.shape) == 0:
        return x.item()
    else:
        return x.detach().cpu().numpy()


# reshape image for visualization
for_viz = lambda x: np.array(
    x.clamp(0, 1).permute(0, 2, 3, 1).detach().cpu().numpy() * 255.0, dtype=np.uint8
)


# Taken from https://github.com/singhgautam/slate/blob/master/slate.py
def visualize(images):
    B, _, H, W = images[0].shape  # first image is observation
    viz_imgs = []
    for _img in images:
        if len(_img.shape) == 4:
            viz_imgs.append(_img)
        else:
            viz_imgs += list(torch.unbind(_img, dim=1))
    viz_imgs = torch.cat(viz_imgs, dim=-1)
    # return torch.cat(torch.unbind(viz_imgs,dim=0), dim=-2).unsqueeze(0)
    return viz_imgs


# Load model and params
def load(model, agent_training=False, resume_checkpoint=None, resume_run_path=None):
    checkpoint = None
    if resume_checkpoint is not None:
        checkpoint = torch.load(
            resume_checkpoint, map_location=next(model._module.parameters()).device
        )
    elif resume_run_path is not None:
        checkpoint = torch.load(
            wandb.restore(
                "checkpoints/model_latest.pth", run_path=resume_run_path
            ).name,
            map_location=next(model._module.parameters()).device,
        )
    else:
        model_checkpoint = Path(wandb.run.dir) / "checkpoints" / "model_latest.pth"
        if model_checkpoint.exists():
            checkpoint = torch.load(
                model_checkpoint, map_location=next(model._module.parameters()).device
            )

    if checkpoint is not None:
        step = checkpoint["step"]
        if agent_training:
            episode = checkpoint["episode"]
        else:
            epoch = checkpoint["epoch"]
            best_val_loss = checkpoint["best_val_loss"]
        model.load(checkpoint)

    else:
        step = 0
        if agent_training:
            episode = 0
        else:
            epoch = 0
            best_val_loss = 1e10

    if agent_training:
        return step, episode
    else:
        return step, epoch, best_val_loss


# Save model and params
def save(
    model,
    step=0,
    epoch=0,
    best_val_loss=1e5,
    episode=0,
    agent_training=False,
    best=False,
):
    sub_dir = "checkpoints"
    model_dir = Path(wandb.run.dir) / sub_dir
    if agent_training:
        checkpoint = {"step": step, "episode": episode}
    else:
        checkpoint = {"step": step, "epoch": epoch, "best_val_loss": best_val_loss}
    checkpoint.update(model.save())
    torch.save(checkpoint, model_dir / f"model_{step}.pth")
    wandb.save(f"{sub_dir}/model_{step}.pth")
    torch.save(checkpoint, model_dir / f"model_latest.pth")
    wandb.save(f"{sub_dir}/model_latest.pth")
    if best:
        torch.save(checkpoint, model_dir / f"model_best.pth")
        wandb.save(f"{sub_dir}/model_best.pth")


# hungarian matching
def hungarian_matching(target, input, return_diff_mat=False):
    tN, tD = target.shape
    iN, iD = input.shape
    assert tN == iN and tD == iD
    diff_mat = np.zeros((tN, iN))
    for t in range(tN):
        for i in range(iN):
            diff_mat[t, i] = torch.norm(target[t] - input[i], p=1).item()
    _, col_ind = linear_sum_assignment(diff_mat)
    if return_diff_mat:
        return torch.LongTensor(col_ind).to(target.device), diff_mat[:, col_ind]
    else:
        return torch.LongTensor(col_ind).to(target.device)


# calculate ARI
def calculate_ari(true_masks, pred_masks):
    true_masks = true_masks.flatten(2)
    pred_masks = pred_masks.flatten(2)

    true_mask_ids = get_item(torch.argmax(true_masks, dim=1))
    pred_mask_ids = get_item(torch.argmax(pred_masks, dim=1))

    aris = []
    for b in range(true_mask_ids.shape[0]):
        aris.append(adjusted_rand_score(true_mask_ids[b], pred_mask_ids[b]))

    return aris


# change img numpy array to torch Tensor
def obs_to_tensor(obs, device):
    if len(obs.shape) == 4:
        return torch.Tensor(obs.transpose(0, 3, 1, 2)).to(device) / 255.0
    else:
        return torch.Tensor(obs).to(device)

class SlotExtractor:
    def __init__(self, model, device, name_model):
        self._model = model
        self._device = device
        self._model.to(device)
        self.name_model = name_model

    def __call__(self, images, prev_slots, to_numpy=True):
        if len(images.shape) == 3:
            batch_images = images[np.newaxis, ...]
        else:
            batch_images = images

        if prev_slots is not None and len(prev_slots.shape) == 2:
            batch_prev_slots = prev_slots[np.newaxis, ...]
        else:
            batch_prev_slots = prev_slots

        batch_images = obs_to_tensor(batch_images, self._device)
        if batch_prev_slots is not None:
            batch_prev_slots = obs_to_tensor(batch_prev_slots, self._device)

        if self.name_model == 'SLATE':
            slots = self._model._module._get_slots(batch_images, prev_slots=batch_prev_slots).detach()
        else:
            slots = self._model(batch_images, prev_slots=batch_prev_slots).detach()

        if len(images.shape) == 3:
            slots = slots[0]

        if to_numpy:
            slots = slots.cpu().numpy()

        return slots

class Dinosaur(torch.nn.Module):
    def __init__(self, dino_model_name, n_slots, slot_dim, intput_feature_dim, num_patches, features, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dino_model_name = dino_model_name
        self._n_slots = n_slots
        self._slot_dim = slot_dim
        self._input_feature_dim = intput_feature_dim
        self._num_patches = num_patches
        self._features = features
        self._normalization = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.feature_extractor = TimmFeatureExtractor(model_name=self._dino_model_name, feature_level=12,
                                                      pretrained=True, freeze=True)
        self.conditioning = RandomConditioning(object_dim=self._slot_dim, n_slots=self._n_slots, learn_mean=True,
                                               learn_std=True)

        pos_embedding = Sequential(DummyPositionEmbed(),
                                   build_two_layer_mlp(input_dim=self._input_feature_dim, output_dim=self._slot_dim,
                                                       hidden_dim=self._input_feature_dim, initial_layer_norm=True))
        ff_mlp = build_two_layer_mlp(input_dim=self._slot_dim, output_dim=self._slot_dim, hidden_dim=4 * self._slot_dim,
                                     initial_layer_norm=True, residual=True)
        self.perceptual_grouping = SlotAttentionGrouping(feature_dim=self._slot_dim, object_dim=self._slot_dim, ff_mlp=ff_mlp,
                                                         positional_embedding=pos_embedding, use_projection_bias=False,
                                                         use_implicit_differentiation=False,
                                                         use_empty_slot_for_masked_slots=False)

        decoder = partial(build_mlp, features=self._features)
        self.object_decoder = PatchDecoder(object_dim=self._slot_dim, output_dim=self._input_feature_dim,
                                           num_patches=self._num_patches, decoder=decoder,)

    def forward(self, image, prev_slots=None):
        image = self._normalization(image)
        feature_extraction_output = self.feature_extractor(image)
        conditioning_output = prev_slots
        if conditioning_output is None:
            conditioning_output = self.conditioning(feature_extraction_output.features.size()[0])

        perceptual_grouping_output = self.perceptual_grouping(feature_extraction_output, conditioning_output)

        return perceptual_grouping_output.objects

    def get_samples(self, image, prev_slots=None):
        image_normalized = self._normalization(image)
        feature_extraction_output = self.feature_extractor(image_normalized)
        conditioning_output = prev_slots
        if conditioning_output is None:
            conditioning_output = self.conditioning(feature_extraction_output.features.size()[0])

        perceptual_grouping_output = self.perceptual_grouping(feature_extraction_output, conditioning_output)
        patch_reconstruction_output = self.object_decoder(perceptual_grouping_output.objects,
                                                          feature_extraction_output.features, image_normalized)
        masks = patch_reconstruction_output.masks_as_image.unsqueeze(1)

        sample = image.unsqueeze(2) * masks + (1 - masks)
        sample = sample.movedim(2, 3).flatten(start_dim=3, end_dim=4)
        sample = torch.cat([image, sample], dim=-1)
        sample = sample.movedim(1, -1).detach().cpu().numpy() * 255
        sample = sample.astype(np.uint8)

        return sample

