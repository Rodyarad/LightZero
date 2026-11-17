## Single-task context windows (cache-free inference)

The single-task `WorldModel` uses three separate lengths, measured in timestep blocks:

| Setting | Meaning |
| --- | --- |
| `policy.num_unroll_steps` | `H_train`: observations/actions passed to the training forward. The policy copies this into `world_model_cfg.train_context_length`. |
| `world_model_cfg.infer_context_length` | `C`: explicit inference window. Roots and next-state policy/value see at most `C` observations and `C-1` preceding actions. Dynamics/reward see at most `C` complete `(state, action)` pairs. |
| `world_model_cfg.max_blocks` | Transformer capacity, independent of the two windows. `max_tokens = tokens_per_block * max_blocks`. |

If `infer_context_length` is `None`, legacy configurations use
`context_length // tokens_per_block` (an exact division is required).
An explicit `infer_context_length` takes precedence over this legacy token budget.
When constructing a model directly, set `train_context_length` explicitly if it differs
from `max_blocks`.

Require `2 <= H_train <= max_blocks` and `1 <= C <= H_train - 1`.
Training keeps its existing teacher-forced layout: the buffer supplies `H_train+1`
observations, the policy drops the last observation, and latent loss supervises
`H_train-1` transitions. Thus dynamics inference must not exceed `H_train-1`
input pairs. Invalid configurations fail immediately rather than silently clipping
the requested window. Padding near episode ends can make the usable training context shorter.

For example, `num_unroll_steps=5, infer_context_length=2, max_blocks=8`
trains on 5 blocks, uses at most 2 blocks during inference, and reserves capacity for 8.
Increasing capacity alone does not increase either window.

Real histories are keyed by `(collect/eval, env_id)` and survive reordered or partial
ready batches. Episode reset clears only the requested IDs in that mode.
Segment-boundary notifications (`reset_init_data=False`) do not clear history.
MCTS appends only the current root-to-leaf branch, never leaves of other simulations.
Different sequence lengths are grouped without padding the transformer input.
Initial/recurrent inference returns compact `output_sequence` features for the current
state, with shape `(B, num_observation_tokens, embed_dim)`; training forward still returns
the full sequence.

These rules apply to the single-task cache-free path. The historical KV-cache/RoPE
description below concerns the older/multi-task implementation and is not the
context-window contract of the single-task path.

## Position Encoding in UniZero World Model

This section provides a detailed explanation of the position encoding strategies used in the UniZero world model and presents two configurable options based on the value of the configuration parameter `self.config.rotary_emb`.

> **Configuration Options:**
> - When `self.config.rotary_emb = False`, **Absolute Position Encoding** (based on `nn.Embedding`) is used.
> - When `self.config.rotary_emb = True`, **Relative Position Encoding** (based on ROPE) is used.

---

### 1. Absolute Position Encoding (Based on `nn.Embedding`)

When the configuration parameter `self.config.rotary_emb` is set to **False**, the model uses `nn.Embedding` for position encoding. The implementation process involves the following steps:

#### 1.1 Embedding Layer Initialization

- **Initialization:**  
  An embedding layer is instantiated using `nn.Embedding`, which maps each position index in the sequence to a fixed-dimensional embedding vector.

#### 1.2 Context Length Restriction

- **kv_cache Management:**  
  Due to the limitation of context length (`context_length`), the model retains only the most recent `<context_length>` steps when caching key-value pairs (kv_cache) to ensure computational efficiency and manageable memory consumption.

#### 1.3 Position Embedding Correction

When reusing the kv_cache, directly utilizing historical position vectors may lead to duplicated or erroneous indices, causing confusion in the model's interpretation of sequence positions. To address this problem, a position embedding correction mechanism is introduced:

- **Problem Description:**  
  Suppose that during inference, the total number of steps is computed as `5 * 2 = 10`, yielding an initial kv_cache with position indices:  
  `0, 1, 2, 3, 4, 5, 6, 7, 8, 9`  
  
  When new data arrives, removing the first 2 steps from the kv_cache leaves:  
  `2, 3, 4, 5, 6, 7, 8, 9`  
  
  If these indices are concatenated directly, it might cause duplicate or incorrect indices, for example:  
  `2, 3, 4, 5, 6, 7, 8, 9, 8, 9`

- **Correction Plan:**  
  To prevent the above issue, the model resets the position indices in the kv_cache to a contiguous sequence, such as:  
  `0, 1, 2, 3, 4, 5, 6, 7`

This mechanism effectively simulates the behavior of relative position encoding, ensuring that errors do not accumulate during kv_cache reuse.

---

### 2. Relative Position Encoding (Based on ROPE)

When the configuration parameter `self.config.rotary_emb` is set to **True**, the model adopts ROPE (Rotary Position Embedding) for position encoding. The main features and implementation process of ROPE are as follows:

#### 2.1 ROPE Initialization

- **Precalculation of Frequency Components:**  
  Frequency components are precalculated and applied to the query and key tensors through a rotational position embedding, directly incorporating positional information into the self-attention computation.

#### 2.2 Episode Time Step-based Indexing

- **Indexing Approach:**  
  Each position index is assigned based on the episode’s time step.  
  For example, when states (`s`) and actions (`a`) alternate, each time step occupies two position indices.  
  Suppose a game consists of 50 steps with states and actions in sequence:  
  `(s₁, a₁, s₂, a₂, ..., s₅₀, a₅₀)`  
  
  The corresponding position indices would be:  
  `1, 2, 3, 4, ..., 99, 100`

#### 2.3 Principles of ROPE

- **Theoretical Basis:**  
  The design of ROPE is inspired by the paper [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864).  
  This method not only encodes absolute positions using a rotation matrix but also directly integrates relative positional information into the self-attention computation, thereby achieving:
  - Greater flexibility (adjustable sequence length);
  - A gradual decay in inter-token dependency with increasing relative distance;
  - Compatibility with relative position encoding in linear self-attention architectures.



#### 3 Performance and Applications

  In environments with shorter dependency relationships (such as Pong or DMC Cartpole-Swingup), the performance of absolute position encoding and ROPE is similar.  
  However, in scenarios that involve longer dependencies, ROPE demonstrates enhanced flexibility and scalability, making it particularly suitable for managing long-range dependency issues in more complex environments.
