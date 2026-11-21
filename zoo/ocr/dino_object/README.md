# DINO object representations from simulator masks

`own_seg_v1` is computed online at reset/step from the simulator's native
segmentation buffer. For each region, the encoder zeros RGB pixels outside the
mask **without cropping or changing the image's spatial layout**, normalizes
the resulting full-size image and runs the frozen pretrained
`vit_small_patch14_dinov2` (DINOv2-S/14). The normalized CLS token yields one
384-dimensional vector per region. If a region is fully occluded, its vector
is zero; the order and shape never change. Masks are privileged simulator
information and must also be available at evaluation time.

Output shapes / order:

- robosuite `Lift`: `(4, 384)` — moving Panda+gripper, walls+floor+mount,
  table, cube.
- ManiSkill `PushCubeCustom-v1`: `(5, 384)` — moving Panda+gripper,
  ground+static base, table, cube, target zone.

Start RL training from the repository root with the existing training entry
pattern, for example:

```
python zoo/robosuite/config/robosuite_soz_segment_config_dino_object.py
python zoo/maniskill/config/maniskill_soz_segment_config_dino_object.py
```

Smaller training configurations, matching the repository's existing
`*_slotcontrast_test.py` variants:

```
python zoo/robosuite/config/robosuite_soz_segment_config_dino_object_test.py
python zoo/maniskill/config/maniskill_soz_segment_config_dino_object_test.py
```

The test configs default to offline Comet logging so an API key is not
required. Set `COMET_START_ONLINE=true` to opt into online logging.
For headless robosuite rendering on Linux use `MUJOCO_GL=egl MPLBACKEND=Agg`.

Both configs use `oc_model_type='DinoObject'` and feed the same slot-shaped
observation to the sampled UniZero world model. The word `slot` in the world
model denotes the input tensor format, not a Slot Attention encoder. Default
DINOv2 weights are managed by timm's Hugging Face cache; optionally specify
`env.dino_weights_path` to load a local state dict.
