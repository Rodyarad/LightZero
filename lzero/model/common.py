"""
Overview:
    In this Python file, we provide a collection of reusable model templates designed to streamline the development
    process for various custom algorithms. By utilizing these pre-built model templates, users can quickly adapt and
    customize their custom algorithms, ensuring efficient and effective development.
    BTW, users can refer to the unittest of these model templates to learn how to use them.
"""
import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from transformers import AutoModelForCausalLM, AutoTokenizer
from ding.torch_utils import MLP, ResBlock
from ding.torch_utils.network.normalization import build_normalization
from ding.utils import SequenceType
from ditk import logging
from ding.utils import set_pkg_seed, get_rank, get_world_size



def MLP_V2(
        in_channels: int,
        hidden_channels: List[int],
        out_channels: int,
        layer_fn: Callable = None,
        activation: Optional[nn.Module] = None,
        norm_type: Optional[str] = None,
        use_dropout: bool = False,
        dropout_probability: float = 0.5,
        output_activation: bool = True,
        output_norm: bool = True,
        last_linear_layer_init_zero: bool = False,
):
    """
    Overview:
        Create a multi-layer perceptron (MLP) using a list of hidden dimensions. Each layer consists of a fully
        connected block with optional activation, normalization, and dropout. The final layer is configurable
        to include or exclude activation, normalization, and dropout based on user preferences.

    Arguments:
        - in_channels (:obj:`int`): Number of input channels (dimensionality of the input tensor).
        - hidden_channels (:obj:`List[int]`): A list specifying the number of channels for each hidden layer.
            For example, [512, 256, 128] means the MLP will have three hidden layers with 512, 256, and 128 units, respectively.
        - out_channels (:obj:`int`): Number of output channels (dimensionality of the output tensor).
        - layer_fn (:obj:`Callable`, optional): Layer function to construct layers (default is `nn.Linear`).
        - activation (:obj:`nn.Module`, optional): Activation function to use after each layer
            (e.g., `nn.ReLU`, `nn.Sigmoid`). Default is None (no activation).
        - norm_type (:obj:`str`, optional): Type of normalization to apply after each layer.
            If None, no normalization is applied. Supported values depend on the implementation of `build_normalization`.
        - use_dropout (:obj:`bool`, optional): Whether to apply dropout after each layer. Default is False.
        - dropout_probability (:obj:`float`, optional): The probability of setting elements to zero in dropout. Default is 0.5.
        - output_activation (:obj:`bool`, optional): Whether to apply activation to the output layer. Default is True.
        - output_norm (:obj:`bool`, optional): Whether to apply normalization to the output layer. Default is True.
        - last_linear_layer_init_zero (:obj:`bool`, optional): Whether to initialize the weights and biases of the
            last linear layer to zeros. This is commonly used in reinforcement learning for stable initial outputs.

    Returns:
        - block (:obj:`nn.Sequential`): A PyTorch `nn.Sequential` object containing the layers of the MLP.

    Notes:
        - The final layer's normalization, activation, and dropout are controlled by `output_activation`,
          `output_norm`, and `use_dropout`.
        - If `last_linear_layer_init_zero` is True, the weights and biases of the last linear layer are initialized to 0.
    """
    assert len(hidden_channels) > 0, "The hidden_channels list must contain at least one element."
    if layer_fn is None:
        layer_fn = nn.Linear

    # Initialize the MLP block
    block = []
    channels = [in_channels] + hidden_channels + [out_channels]

    # Build all layers except the final layer
    for i, (in_channels, out_channels) in enumerate(zip(channels[:-2], channels[1:-1])):
        block.append(layer_fn(in_channels, out_channels))
        if norm_type is not None:
            block.append(build_normalization(norm_type, dim=1)(out_channels))
        if activation is not None:
            block.append(activation)
        if use_dropout:
            block.append(nn.Dropout(dropout_probability))

    # Build the final layer
    in_channels = channels[-2]
    out_channels = channels[-1]
    block.append(layer_fn(in_channels, out_channels))

    # Add optional normalization and activation for the final layer
    if output_norm and norm_type is not None:
        block.append(build_normalization(norm_type, dim=1)(out_channels))
    if output_activation and activation is not None:
        block.append(activation)
    if use_dropout:
        block.append(nn.Dropout(dropout_probability))

    # Initialize the weights and biases of the last linear layer to zero if specified
    if last_linear_layer_init_zero:
        for layer in reversed(block):
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
                break

    return nn.Sequential(*block)

# use dataclass to make the output of network more convenient to use
@dataclass
class MZRNNNetworkOutput:
    # output format of the MuZeroRNN model
    value: torch.Tensor
    value_prefix: torch.Tensor
    policy_logits: torch.Tensor
    latent_state: torch.Tensor
    predict_next_latent_state: torch.Tensor
    reward_hidden_state: Tuple[torch.Tensor]


@dataclass
class EZNetworkOutput:
    # output format of the EfficientZero model
    value: torch.Tensor
    value_prefix: torch.Tensor
    policy_logits: torch.Tensor
    latent_state: torch.Tensor
    reward_hidden_state: Tuple[torch.Tensor]


@dataclass
class MZNetworkOutput:
    # output format of the MuZero model
    value: torch.Tensor
    reward: torch.Tensor
    policy_logits: torch.Tensor
    latent_state: torch.Tensor


class SimNorm(nn.Module):

    def __init__(self, simnorm_dim: int) -> None:
        """
        Overview:
            Simplicial normalization. Adapted from https://arxiv.org/abs/2204.00616.
        Arguments:
            - simnorm_dim (:obj:`int`): The dimension for simplicial normalization.
        """
        super().__init__()
        self.dim = simnorm_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Forward pass of the SimNorm layer.
        Arguments:
            - x (:obj:`torch.Tensor`): The input tensor to normalize.
        Returns:
            - x (:obj:`torch.Tensor`): The normalized tensor.
        """
        shp = x.shape
        # Ensure that there is at least one simplex to normalize across.
        if shp[1] != 0:
            x = x.view(*shp[:-1], -1, self.dim)
            x = F.softmax(x, dim=-1)
            return x.view(*shp)
        else:
            return x

    def __repr__(self) -> str:
        """
        Overview:
            String representation of the SimNorm layer.
        Returns:
            - output (:obj:`str`): The string representation.
        """
        return f"SimNorm(dim={self.dim})"


def AvgL1Norm(x, eps=1e-8):
    """
    Overview:
        Normalize the input tensor by the L1 norm.
    Arguments:
        - x (:obj:`torch.Tensor`): The input tensor to normalize.
        - eps (:obj:`float`): The epsilon value to prevent division by zero.
    Returns:
        - :obj:`torch.Tensor`: The normalized tensor.
    """
    return x / x.abs().mean(-1, keepdim=True).clamp(min=eps)


class FeatureAndGradientHook:

    def __init__(self):
        """
        Overview:
            Class to capture features and gradients at SimNorm.
        """
        self.features_before = []
        self.features_after = []
        self.grads_before = []
        self.grads_after = []

    def setup_hooks(self, model):
        # Hooks to capture features and gradients at SimNorm
        self.forward_handler = model.sim_norm.register_forward_hook(self.forward_hook)
        self.backward_handler = model.sim_norm.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        with torch.no_grad():
            self.features_before.append(input[0])
            self.features_after.append(output)

    def backward_hook(self, module, grad_input, grad_output):
        with torch.no_grad():
            self.grads_before.append(grad_input[0] if grad_input[0] is not None else None)
            self.grads_after.append(grad_output[0] if grad_output[0] is not None else None)

    def analyze(self):
        # Calculate L2 norms of features
        l2_norm_before = torch.mean(torch.stack([torch.norm(f, p=2, dim=1).mean() for f in self.features_before]))
        l2_norm_after = torch.mean(torch.stack([torch.norm(f, p=2, dim=1).mean() for f in self.features_after]))

        # Calculate norms of gradients
        grad_norm_before = torch.mean(
            torch.stack([torch.norm(g, p=2, dim=1).mean() for g in self.grads_before if g is not None]))
        grad_norm_after = torch.mean(
            torch.stack([torch.norm(g, p=2, dim=1).mean() for g in self.grads_after if g is not None]))

        # Clear stored data and delete tensors to free memory
        self.clear_data()

        # Optionally clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return l2_norm_before, l2_norm_after, grad_norm_before, grad_norm_after

    def clear_data(self):
        del self.features_before[:]
        del self.features_after[:]
        del self.grads_before[:]
        del self.grads_after[:]

    def remove_hooks(self):
        self.forward_handler.remove()
        self.backward_handler.remove()


class DownSample(nn.Module):

    def __init__(self, observation_shape: SequenceType, out_channels: int,
                 activation: nn.Module = nn.ReLU(inplace=True),
                 norm_type: Optional[str] = 'BN',
                 num_resblocks: int = 1,
                 ) -> None:
        """
        Overview:
            Define downSample convolution network. Encode the observation into hidden state.
            This network is often used in video games like Atari. In board games like go and chess,
            we don't need this module.
        Arguments:
            - observation_shape (:obj:`SequenceType`): The shape of observation space, e.g. [C, W, H]=[12, 96, 96]
                for video games like atari, RGB 3 channel times stack 4 frames.
            - out_channels (:obj:`int`): The output channels of output hidden state.
            - activation (:obj:`nn.Module`): The activation function used in network, defaults to nn.ReLU(inplace=True). \
                Use the inplace operation to speed up.
            - norm_type (:obj:`Optional[str]`): The normalization type used in network, defaults to 'BN'.
            - num_resblocks (:obj:`int`): The number of residual blocks. Defaults to 1.
        """
        super().__init__()
        assert norm_type in ['BN', 'LN'], "norm_type must in ['BN', 'LN']"

        self.observation_shape = observation_shape
        self.conv1 = nn.Conv2d(
            observation_shape[0],
            out_channels // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,  # disable bias for better convergence
        )
        if norm_type == 'BN':
            self.norm1 = nn.BatchNorm2d(out_channels // 2)
        elif norm_type == 'LN':
            self.norm1 = nn.LayerNorm([out_channels // 2, observation_shape[-2] // 2, observation_shape[-1] // 2],
                                      eps=1e-5)

        self.resblocks1 = nn.ModuleList(
            [
                ResBlock(
                    in_channels=out_channels // 2,
                    activation=activation,
                    norm_type=norm_type,
                    res_type='basic',
                    bias=False
                ) for _ in range(num_resblocks)
            ]
        )
        self.downsample_block = ResBlock(
            in_channels=out_channels // 2,
            out_channels=out_channels,
            activation=activation,
            norm_type=norm_type,
            res_type='downsample',
            bias=False
        )
        self.resblocks2 = nn.ModuleList(
            [
                ResBlock(
                    in_channels=out_channels, activation=activation, norm_type=norm_type, res_type='basic', bias=False
                ) for _ in range(num_resblocks)
            ]
        )
        self.pooling1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.resblocks3 = nn.ModuleList(
            [
                ResBlock(
                    in_channels=out_channels, activation=activation, norm_type=norm_type, res_type='basic', bias=False
                ) for _ in range(1)
            ]
        )
        self.pooling2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shapes:
            - x (:obj:`torch.Tensor`): :math:`(B, C_in, W, H)`, where B is batch size, C_in is channel, W is width, \
                H is height.
            - output (:obj:`torch.Tensor`): :math:`(B, C_out, W_, H_)`, where B is batch size, C_out is channel, W_ is \
                output width, H_ is output height.
        """
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)

        for block in self.resblocks1:
            x = block(x)
        x = self.downsample_block(x)
        for block in self.resblocks2:
            x = block(x)
        x = self.pooling1(x)
        for block in self.resblocks3:
            x = block(x)

        # 64, 84, 96 are the most common observation shapes in Atari games.
        if self.observation_shape[1] == 64:
            output = x
        elif self.observation_shape[1] == 84:
            x = self.pooling2(x)
            output = x
        elif self.observation_shape[1] == 96:
            x = self.pooling2(x)
            output = x
        else:
            raise NotImplementedError(f"DownSample for observation shape {self.observation_shape} is not implemented now. "
                                      f"You should transform the observation shape to 64 or 96 in the env.")

        return output

class QwenNetwork(nn.Module):
    def __init__(self,
                 model_path: str = 'Qwen/Qwen3-1.7B',
                 embedding_size: int = 768,
                 final_norm_option_in_encoder: str = "layernorm",
                 group_size: int = 8,
                 tokenizer=None):
        super().__init__()

        logging.info(f"Loading Qwen model from: {model_path}")
        
        local_rank = get_rank()
        if local_rank == 0:
            self.pretrained_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype="auto", 
                device_map={"": local_rank},
                attn_implementation="flash_attention_2"
            )
        if get_world_size() > 1:
            torch.distributed.barrier()
        if local_rank != 0:
            self.pretrained_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map={"": local_rank}, 
                attn_implementation="flash_attention_2"
            )
        
        for p in self.pretrained_model.parameters():
            p.requires_grad = False

        if tokenizer is None:
            if local_rank == 0:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            if get_world_size() > 1:
                torch.distributed.barrier()
            if local_rank != 0:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            self.tokenizer = tokenizer

        qwen_hidden_size = self.pretrained_model.config.hidden_size

        self.embedding_head = nn.Sequential(
            nn.Linear(qwen_hidden_size, embedding_size),
            self._create_norm_layer(final_norm_option_in_encoder, embedding_size, group_size)
        )

    def _create_norm_layer(self, norm_option, embedding_size, group_size):
        if norm_option.lower() == "simnorm":
            return SimNorm(simnorm_dim=group_size)
        elif norm_option.lower() == "layernorm":
            return nn.LayerNorm(embedding_size)
        else:
            raise NotImplementedError(f"Normalization type '{norm_option}' is not implemented.")

    def encode(self, x: torch.Tensor, no_grad: bool = True) -> torch.Tensor:
        """
        Overview:
            Encode the input token sequence `x` into a latent representation
            using a pretrained language model backbone followed by a projection head.
        Arguments:
            - x (:obj:`torch.Tensor`):  Input token ids of shape (B, L)
            - no_grad (:obj:`bool`, optional, default=True): If True, encoding is performed under `torch.no_grad()` to save memory and computation (no gradient tracking).
        Returns:
            - latent (:obj:`torch.Tensor`): Encoded latent state of shape (B, D).
        """
        pad_id = self.tokenizer.pad_token_id
        attention_mask = (x != pad_id).long().to(x.device)
        context = {'input_ids': x.long(), 'attention_mask': attention_mask}
        if no_grad:
            with torch.no_grad():
                outputs = self.pretrained_model(**context, output_hidden_states=True, return_dict=True)
        else:
            outputs = self.pretrained_model(**context, output_hidden_states=True, return_dict=True)
        last_hidden = outputs.hidden_states[-1]

        B, L, H = last_hidden.size()
        lengths = attention_mask.sum(dim=1)  # [B]
        positions = torch.clamp(lengths - 1, min=0)  # [B]
        batch_idx = torch.arange(B, device=last_hidden.device)

        selected = last_hidden[batch_idx, positions]  # [B, H]

        latent = self.embedding_head(selected.to(self.embedding_head[0].weight.dtype))
        return latent

    def decode(self, embeddings: torch.Tensor, max_length: int = 512) -> str:
        """
        Decodes embeddings into text via the decoder network.
        """
        embeddings_detached = embeddings.detach()
        self.pretrained_model.eval()
        
        # Directly generate using provided embeddings
        with torch.no_grad():
            param = next(self.pretrained_model.parameters())
            embeddings = embeddings_detached.to(device=param.device, dtype=param.dtype)
            gen_ids = self.pretrained_model.generate(
                inputs_embeds=embeddings,
                max_length=max_length
            )
        texts = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        self.pretrained_model.train()
        return texts[0] if len(texts) == 1 else texts

    def forward(self, x: torch.Tensor, no_grad: bool = True) -> torch.Tensor:
        return self.encode(x, no_grad=no_grad)


class HFLanguageRepresentationNetwork(nn.Module):
    def __init__(self,
                model_path: str = 'google-bert/bert-base-uncased',
                embedding_size: int = 768,
                group_size: int = 8,
                final_norm_option_in_encoder: str = "layernorm",
                tokenizer=None):
        """
        Overview:
            This class defines a language representation network that utilizes a pretrained Hugging Face model.
            The network outputs embeddings with the specified dimension and can optionally use SimNorm or LayerNorm
            for normalization at the final stage to ensure training stability.
        Arguments:
            - model_path (str): The path to the pretrained Hugging Face model. Default is 'google-bert/bert-base-uncased'.
            - embedding_size (int): The dimension of the output embeddings. Default is 768.
            - group_size (int): The group size for SimNorm when using normalization.
            - final_norm_option_in_encoder (str): The type of normalization to use ("simnorm" or "layernorm"). Default is "layernorm".
            - tokenizer (Optional): An instance of a tokenizer. If None, the tokenizer will be loaded from the pretrained model.
        """
        super().__init__()

        from transformers import AutoModel, AutoTokenizer
        logging.info(f"Loading model from: {model_path}")

        # In distributed training, only the rank 0 process downloads the model, and other processes load from cache to speed up startup.
        if get_rank() == 0:
            self.pretrained_model = AutoModel.from_pretrained(model_path)

        if get_world_size() > 1:
            # Wait for rank 0 to finish loading the model.
            torch.distributed.barrier()
        if get_rank() != 0:
            self.pretrained_model = AutoModel.from_pretrained(model_path)

        if tokenizer is None:
            # Only rank 0 downloads the tokenizer, and then other processes load it from cache.
            if get_rank() == 0:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            if get_world_size() > 1:
                torch.distributed.barrier()
            if get_rank() != 0:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            self.tokenizer = tokenizer

        # Set the embedding dimension. A linear projection is added (the dimension remains unchanged here but can be extended for other mappings).
        self.embedding_size = embedding_size
        self.embed_proj_head = nn.Linear(self.pretrained_model.config.hidden_size, self.embedding_size)

        # # Select the normalization method based on the final_norm_option_in_encoder parameter.
        if final_norm_option_in_encoder.lower() == "simnorm":
            self.norm = SimNorm(simnorm_dim=group_size)
        elif final_norm_option_in_encoder.lower() == "layernorm":
            self.norm = nn.LayerNorm(embedding_size)
        else:
            raise NotImplementedError(f"Normalization type '{final_norm_option_in_encoder}' is not implemented. "
                                      f"Choose 'simnorm' or 'layernorm'.")

    def forward(self, x: torch.Tensor, no_grad: bool = True) -> torch.Tensor:
        """
        Forward Propagation:
            Compute the language representation based on the input token sequence.
            The [CLS] token’s representation is extracted from the output of the pretrained model,
            then passed through a linear projection and final normalization layer (SimNorm or LayerNorm).

        Arguments:
            - x (torch.Tensor): Input token sequence of shape [batch_size, seq_len].
            - no_grad (bool): Whether to run in no-gradient mode for memory efficiency. Default is True.
        Returns:
        - torch.Tensor: The processed language embedding with shape [batch_size, embedding_size].
        """

        # Construct the attention mask to exclude padding tokens.
        attention_mask = x != self.tokenizer.pad_token_id

        # Use no_grad context if specified to disable gradient computation.
        if no_grad:
            with torch.no_grad():
                x = x.long()  # Ensure the input tensor is of type long.
                outputs = self.pretrained_model(x, attention_mask=attention_mask)
                # Get the hidden state from the last layer and select the output corresponding to the [CLS] token.
                cls_embedding = outputs.last_hidden_state[:, 0, :]
        else:
            x = x.long()
            outputs = self.pretrained_model(x, attention_mask=attention_mask)
            cls_embedding = outputs.last_hidden_state[:, 0, :]

        # Apply linear projection to obtain the desired output dimension.
        cls_embedding = self.embed_proj_head(cls_embedding)
        # Normalize the embeddings using the selected normalization layer (SimNorm or LayerNorm) to ensure training stability.
        cls_embedding = self.norm(cls_embedding)
        
        return cls_embedding


class RepresentationNetworkUniZero(nn.Module):
    
    def __init__(
            self,
            observation_shape: SequenceType = (3, 64, 64),
            num_res_blocks: int = 1,
            num_channels: int = 64,
            downsample: bool = True,
            activation: nn.Module = nn.GELU(approximate='tanh'),
            norm_type: str = 'BN',
            embedding_dim: int = 256,
            group_size: int = 8,
            final_norm_option_in_encoder: str = 'LayerNorm', # TODO
    ) -> None:
        """
        Overview:
            Representation network used in UniZero. Encode the 2D image obs into latent state.
            Currently, the network only supports obs images with both a width and height of 64.
        Arguments:
            - observation_shape (:obj:`SequenceType`): The shape of observation space, e.g. [C, W, H]=[3, 64, 64]
                for video games like atari, RGB 3 channel.
            - num_res_blocks (:obj:`int`): The number of residual blocks.
            - num_channels (:obj:`int`): The channel of output hidden state.
            - downsample (:obj:`bool`): Whether to do downsampling for observations in ``representation_network``, \
                defaults to True. This option is often used in video games like Atari. In board games like go, \
                we don't need this module.
            - activation (:obj:`nn.Module`): The activation function used in network, defaults to nn.ReLU(inplace=True). \
                Use the inplace operation to speed up.
            - norm_type (:obj:`str`): The type of normalization in networks. defaults to 'BN'.
            - embedding_dim (:obj:`int`): The dimension of the latent state.
            - group_size (:obj:`int`): The dimension for simplicial normalization.
            - final_norm_option_in_encoder (:obj:`str`): The normalization option for the final layer, defaults to 'SimNorm'. \
                Options are 'SimNorm' and 'LayerNorm'.
        """
        super().__init__()
        assert norm_type in ['BN', 'LN'], "norm_type must in ['BN', 'LN']"
        logging.info(f"Using norm type: {norm_type}")
        logging.info(f"Using activation type: {activation}")

        self.observation_shape = observation_shape
        self.downsample = downsample
        if self.downsample:
            self.downsample_net = DownSample(
                observation_shape,
                num_channels,
                activation=activation,
                norm_type=norm_type,
            )
        else:
            self.conv = nn.Conv2d(observation_shape[0], num_channels, kernel_size=3, stride=1, padding=1, bias=False)

            if norm_type == 'BN':
                self.norm = nn.BatchNorm2d(num_channels)
            elif norm_type == 'LN':
                if downsample:
                    self.norm = nn.LayerNorm(
                        [num_channels, math.ceil(observation_shape[-2] / 16), math.ceil(observation_shape[-1] / 16)],
                        eps=1e-5)
                else:
                    self.norm = nn.LayerNorm([num_channels, observation_shape[-2], observation_shape[-1]], eps=1e-5)

        self.resblocks = nn.ModuleList(
            [
                ResBlock(
                    in_channels=num_channels, activation=activation, norm_type=norm_type, res_type='basic', bias=False
                ) for _ in range(num_res_blocks)
            ]
        )
        self.activation = activation
        self.embedding_dim = embedding_dim

        if self.observation_shape[1] == 64:
            self.last_linear = nn.Linear(64 * 8 * 8, self.embedding_dim, bias=False)

        elif self.observation_shape[1] in [84, 96]:
            self.last_linear = nn.Linear(64 * 6 * 6, self.embedding_dim, bias=False)

        self.final_norm_option_in_encoder = final_norm_option_in_encoder
        if self.final_norm_option_in_encoder == 'LayerNorm':
            self.final_norm = nn.LayerNorm(self.embedding_dim, eps=1e-5)
        elif self.final_norm_option_in_encoder == 'SimNorm':
            self.final_norm = SimNorm(simnorm_dim=group_size)
        else:
            raise ValueError(f"Unsupported final_norm_option_in_encoder: {self.final_norm_option_in_encoder}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shapes:
            - x (:obj:`torch.Tensor`): :math:`(B, C_in, W, H)`, where B is batch size, C_in is channel, W is width, \
                H is height.
            - output (:obj:`torch.Tensor`): :math:`(B, C_out, W_, H_)`, where B is batch size, C_out is channel, W_ is \
                output width, H_ is output height.
        """
        if self.downsample:
            x = self.downsample_net(x)
        else:
            x = self.conv(x)
            x = self.norm(x)
            x = self.activation(x)
        for block in self.resblocks:
            x = block(x)

        # Important: Transform the output feature plane to the latent state.
        # For example, for an Atari feature plane of shape (64, 8, 8),
        # flattening results in a size of 4096, which is then transformed to 768.
        x = self.last_linear(x.view(x.size(0), -1))

        x = x.view(-1, self.embedding_dim)

        # NOTE: very important for training stability.
        x = self.final_norm(x)

        return x


class RepresentationNetwork(nn.Module):

    def __init__(
            self,
            observation_shape: SequenceType = (4, 96, 96),
            num_res_blocks: int = 1,
            num_channels: int = 64,
            downsample: bool = True,
            activation: nn.Module = nn.ReLU(inplace=True),
            norm_type: str = 'BN',
            embedding_dim: int = 256,
            group_size: int = 8,
            use_sim_norm: bool = False,
    ) -> None:
        """
        Overview:
            Representation network used in MuZero and derived algorithms. Encode the 2D image obs into latent state.
            Currently, the network only supports obs images with both a width and height of 96.
        Arguments:
            - observation_shape (:obj:`SequenceType`): The shape of observation space, e.g. [C, W, H]=[4, 96, 96]
                for video games like atari, 1 gray channel times stack 4 frames.
            - num_res_blocks (:obj:`int`): The number of residual blocks.
            - num_channels (:obj:`int`): The channel of output hidden state.
            - downsample (:obj:`bool`): Whether to do downsampling for observations in ``representation_network``, \
                defaults to True. This option is often used in video games like Atari. In board games like go, \
                we don't need this module.
            - activation (:obj:`nn.Module`): The activation function used in network, defaults to nn.ReLU(inplace=True). \
                Use the inplace operation to speed up.
            - norm_type (:obj:`str`): The type of normalization in networks. defaults to 'BN'.
            - embedding_dim (:obj:`int`): The dimension of the output hidden state.
            - group_size (:obj:`int`): The size of group in the SimNorm layer.
            - use_sim_norm (:obj:`bool`): Whether to use SimNorm layer, defaults to False.
        """
        super().__init__()
        assert norm_type in ['BN', 'LN'], "norm_type must in ['BN', 'LN']"

        self.downsample = downsample
        if self.downsample:
            self.downsample_net = DownSample(
                observation_shape,
                num_channels,
                activation=activation,
                norm_type=norm_type,
            )
        else:
            self.conv = nn.Conv2d(observation_shape[0], num_channels, kernel_size=3, stride=1, padding=1, bias=False)

            if norm_type == 'BN':
                self.norm = nn.BatchNorm2d(num_channels)
            elif norm_type == 'LN':
                if downsample:
                    self.norm = nn.LayerNorm(
                        [num_channels, math.ceil(observation_shape[-2] / 16), math.ceil(observation_shape[-1] / 16)],
                        eps=1e-5)
                else:
                    self.norm = nn.LayerNorm([num_channels, observation_shape[-2], observation_shape[-1]], eps=1e-5)

        self.resblocks = nn.ModuleList(
            [
                ResBlock(
                    in_channels=num_channels, activation=activation, norm_type=norm_type, res_type='basic', bias=False
                ) for _ in range(num_res_blocks)
            ]
        )
        self.activation = activation

        self.use_sim_norm = use_sim_norm

        if self.use_sim_norm:
            self.embedding_dim = embedding_dim
            self.sim_norm = SimNorm(simnorm_dim=group_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shapes:
            - x (:obj:`torch.Tensor`): :math:`(B, C_in, W, H)`, where B is batch size, C_in is channel, W is width, \
                H is height.
            - output (:obj:`torch.Tensor`): :math:`(B, C_out, W_, H_)`, where B is batch size, C_out is channel, W_ is \
                output width, H_ is output height.
        """
        if self.downsample:
            x = self.downsample_net(x)
        else:
            x = self.conv(x)
            x = self.norm(x)
            x = self.activation(x)

        for block in self.resblocks:
            x = block(x)

        if self.use_sim_norm:
            # NOTE: very important. 
            # for atari 64,8,8 = 4096 -> 768
            x = self.sim_norm(x)

        return x


class RepresentationNetworkMLP(nn.Module):

    def __init__(
            self,
            observation_shape: int,
            hidden_channels: int = 64,
            layer_num: int = 2,
            activation: nn.Module = nn.GELU(approximate='tanh'),
            norm_type: Optional[str] = 'BN',
            group_size: int = 8,
            final_norm_option_in_encoder: str = 'LayerNorm', # TODO
    ) -> torch.Tensor:
        """
        Overview:
            Representation network used in MuZero and derived algorithms. Encode the vector obs into latent state \
                with Multi-Layer Perceptron (MLP).
        Arguments:
            - observation_shape (:obj:`int`): The shape of vector observation space, e.g. N = 10.
            - num_res_blocks (:obj:`int`): The number of residual blocks.
            - hidden_channels (:obj:`int`): The channel of output hidden state.
            - downsample (:obj:`bool`): Whether to do downsampling for observations in ``representation_network``, \
                defaults to True. This option is often used in video games like Atari. In board games like go, \
                we don't need this module.
            - activation (:obj:`nn.Module`): The activation function used in network, defaults to nn.ReLU(inplace=True). \
                Use the inplace operation to speed up.
            - norm_type (:obj:`str`): The type of normalization in networks. defaults to 'BN'.
        """
        super().__init__()
        self.fc_representation = MLP(
            in_channels=observation_shape,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            layer_num=layer_num,
            activation=activation,
            norm_type=norm_type,
            # don't use activation and norm in the last layer of representation network is important for convergence.
            output_activation=False,
            output_norm=False,
            # last_linear_layer_init_zero=True is beneficial for convergence speed.
            last_linear_layer_init_zero=True,
        )

        # # Select the normalization method based on the final_norm_option_in_encoder parameter.
        if final_norm_option_in_encoder.lower() == "simnorm":
            self.norm = SimNorm(simnorm_dim=group_size)
        elif final_norm_option_in_encoder.lower() == "layernorm":
            self.norm = nn.LayerNorm(hidden_channels)
        else:
            raise NotImplementedError(f"Normalization type '{final_norm_option_in_encoder}' is not implemented. "
                                      f"Choose 'simnorm' or 'layernorm'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shapes:
            - x (:obj:`torch.Tensor`): :math:`(B, N)`, where B is batch size, N is the length of vector observation.
            - output (:obj:`torch.Tensor`): :math:`(B, hidden_channels)`, where B is batch size.
        """
        x = self.fc_representation(x)
        x = self.norm(x)

        return x


class LatentDecoder(nn.Module):
    
    def __init__(self, embedding_dim: int, output_shape: SequenceType, num_channels: int = 64, activation: nn.Module = nn.GELU(approximate='tanh')):
        """
        Overview:
            Decoder network used in UniZero. Decode the latent state into 2D image obs.
        Arguments:
            - embedding_dim (:obj:`int`): The dimension of the latent state.
            - output_shape (:obj:`SequenceType`): The shape of observation space, e.g. [C, W, H]=[3, 64, 64]
                for video games like atari, RGB 3 channel times stack 4 frames.
            - num_channels (:obj:`int`): The channel of output hidden state.
            - activation (:obj:`nn.Module`): The activation function used in network, defaults to nn.GELU(approximate='tanh').
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.output_shape = output_shape  # (C, H, W)
        self.num_channels = num_channels
        self.activation = activation

        # Assuming that the output shape is (C, H, W) = (12, 96, 96) and embedding_dim is 256
        # We will reverse the process of the representation network
        self.initial_size = (
            num_channels, output_shape[1] // 8, output_shape[2] // 8)  # This should match the last layer of the encoder
        self.fc = nn.Linear(self.embedding_dim, np.prod(self.initial_size))

        # Upsampling blocks
        self.conv_blocks = nn.ModuleList([
            # Block 1: (num_channels, H/8, W/8) -> (num_channels//2, H/4, W/4)
            nn.ConvTranspose2d(num_channels, num_channels // 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            self.activation,
            nn.BatchNorm2d(num_channels // 2),
            # Block 2: (num_channels//2, H/4, W/4) -> (num_channels//4, H/2, W/2)
            nn.ConvTranspose2d(num_channels // 2, num_channels // 4, kernel_size=3, stride=2, padding=1,
                               output_padding=1),
            self.activation,
            nn.BatchNorm2d(num_channels // 4),
            # Block 3: (num_channels//4, H/2, W/2) -> (output_shape[0], H, W)
            nn.ConvTranspose2d(num_channels // 4, output_shape[0], kernel_size=3, stride=2, padding=1,
                               output_padding=1),
        ])
        # TODO: last layer use sigmoid?

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        # Map embeddings back to the image space
        x = self.fc(embeddings)  # (B, embedding_dim) -> (B, C*H/8*W/8)
        x = x.view(-1, *self.initial_size)  # (B, C*H/8*W/8) -> (B, C, H/8, W/8)

        # Apply conv blocks
        for block in self.conv_blocks:
            x = block(x)  # Upsample progressively

        # The output x should have the shape of (B, output_shape[0], output_shape[1], output_shape[2])
        return x


class LatentEncoderForMemoryEnv(nn.Module):

    def __init__(
            self,
            image_shape=(3, 5, 5),
            embedding_size=100,
            channels=[16, 32, 64],
            kernel_sizes=[3, 3, 3],
            strides=[1, 1, 1],
            activation: nn.Module = nn.GELU(approximate='tanh'),
            normalize_pixel=False,
            group_size: int = 8,
            **kwargs,
    ):
        """
        Overview:
            Encoder network used in UniZero in MemoryEnv. Encode the 2D image obs into latent state.
        Arguments:
            - image_shape (:obj:`SequenceType`): The shape of observation space, e.g. [C, W, H]=[3, 64, 64]
                for video games like atari, RGB 3 channel times stack 4 frames.
            - embedding_size (:obj:`int`): The dimension of the latent state.
            - channels (:obj:`List[int]`): The channel of output hidden state.
            - kernel_sizes (:obj:`List[int]`): The kernel size of convolution layers.
            - strides (:obj:`List[int]`): The stride of convolution layers.
            - activation (:obj:`nn.Module`): The activation function used in network, defaults to nn.GELU(approximate='tanh'). \
                Use the inplace operation to speed up.
            - normalize_pixel (:obj:`bool`): Whether to normalize the pixel values to [0, 1], defaults to False.
            - group_size (:obj:`int`): The dimension for simplicial normalization
        """
        super(LatentEncoderForMemoryEnv, self).__init__()
        self.shape = image_shape
        self.channels = [image_shape[0]] + list(channels)

        layers = []
        for i in range(len(self.channels) - 1):
            layers.append(
                nn.Conv2d(
                    self.channels[i], self.channels[i + 1], kernel_sizes[i], strides[i],
                    padding=kernel_sizes[i] // 2  # keep the same size of feature map
                )
            )
            layers.append(nn.BatchNorm2d(self.channels[i + 1]))
            layers.append(activation)

        layers.append(nn.AdaptiveAvgPool2d(1))

        self.cnn = nn.Sequential(*layers)
        self.linear = nn.Sequential(
            nn.Linear(self.channels[-1], embedding_size, bias=False),
        )
        init.kaiming_normal_(self.linear[0].weight, mode='fan_out', nonlinearity='relu')

        self.normalize_pixel = normalize_pixel
        self.sim_norm = SimNorm(simnorm_dim=group_size)

    def forward(self, image):
        if self.normalize_pixel:
            image = image / 255.0
        x = self.cnn(image.float())  # (B, C, 1, 1)
        x = torch.flatten(x, start_dim=1)  # (B, C)
        x = self.linear(x)  # (B, embedding_size)
        x = self.sim_norm(x)
        return x


class LatentDecoderForMemoryEnv(nn.Module):

    def __init__(
            self,
            image_shape=(3, 5, 5),
            embedding_size=256,
            channels=[64, 32, 16],
            kernel_sizes=[3, 3, 3],
            strides=[1, 1, 1],
            activation: nn.Module = nn.LeakyReLU(negative_slope=0.01),
            **kwargs,
    ):
        """
        Overview:
            Decoder network used in UniZero in MemoryEnv. Decode the latent state into 2D image obs.
        Arguments:
            - image_shape (:obj:`SequenceType`): The shape of observation space, e.g. [C, W, H]=[3, 64, 64]
                for video games like atari, RGB 3 channel times stack 4 frames.
            - embedding_size (:obj:`int`): The dimension of the latent state.
            - channels (:obj:`List[int]`): The channel of output hidden state.
            - kernel_sizes (:obj:`List[int]`): The kernel size of convolution layers.
            - strides (:obj:`List[int]`): The stride of convolution layers.
            - activation (:obj:`nn.Module`): The activation function used in network, defaults to nn.LeakyReLU(). \
                Use the inplace operation to speed up.
        """
        super(LatentDecoderForMemoryEnv, self).__init__()
        self.shape = image_shape
        self.channels = list(channels) + [image_shape[0]]

        self.linear = nn.Linear(embedding_size, channels[0] * image_shape[1] * image_shape[2])

        layers = []
        for i in range(len(self.channels) - 1):
            layers.append(
                nn.ConvTranspose2d(
                    self.channels[i], self.channels[i + 1], kernel_sizes[i], strides[i],
                    padding=kernel_sizes[i] // 2, output_padding=strides[i] - 1
                )
            )
            if i < len(self.channels) - 2:
                layers.append(nn.BatchNorm2d(self.channels[i + 1]))
                layers.append(activation)
            else:
                layers.append(nn.Sigmoid())

        self.deconv = nn.Sequential(*layers)

    def forward(self, embedding):
        x = self.linear(embedding)
        x = x.view(-1, self.channels[0], self.shape[1], self.shape[2])
        x = self.deconv(x)  # (B, C, H, W)
        return x


class VectorDecoderForMemoryEnv(nn.Module):

    def __init__(
            self,
            embedding_dim: int,
            output_shape: SequenceType,
            hidden_channels: int = 64,
            layer_num: int = 2,
            activation: nn.Module = nn.LeakyReLU(negative_slope=0.01),  # TODO
            norm_type: Optional[str] = 'BN',
    ) -> torch.Tensor:
        """
        Overview:
            Decoder network used in UniZero in MemoryEnv. Decode the latent state into vector obs.
        Arguments:
            - observation_shape (:obj:`int`): The shape of vector observation space, e.g. N = 10.
            - num_res_blocks (:obj:`int`): The number of residual blocks.
            - hidden_channels (:obj:`int`): The channel of output hidden state.
            - downsample (:obj:`bool`): Whether to do downsampling for observations in ``representation_network``, \
                defaults to True. This option is often used in video games like Atari. In board games like go, \
                we don't need this module.
            - activation (:obj:`nn.Module`): The activation function used in network, defaults to nn.ReLU(). \
                Use the inplace operation to speed up.
            - norm_type (:obj:`str`): The type of normalization in networks. defaults to 'BN'.
        """
        super().__init__()
        self.fc_representation = MLP(
            in_channels=embedding_dim,
            hidden_channels=hidden_channels,
            out_channels=output_shape,
            layer_num=layer_num,
            activation=activation,
            norm_type=norm_type,
            # don't use activation and norm in the last layer of representation network is important for convergence.
            output_activation=False,
            output_norm=False,
            # last_linear_layer_init_zero=True is beneficial for convergence speed.
            last_linear_layer_init_zero=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shapes:
            - x (:obj:`torch.Tensor`): :math:`(B, N)`, where B is batch size, N is the length of vector observation.
            - output (:obj:`torch.Tensor`): :math:`(B, hidden_channels)`, where B is batch size.
        """
        x = self.fc_representation(x)
        return x


class PredictionNetwork(nn.Module):

    def __init__(
            self,
            observation_shape: SequenceType,
            action_space_size: int,
            num_res_blocks: int,
            num_channels: int,
            value_head_channels: int,
            policy_head_channels: int,
            value_head_hidden_channels: int,
            policy_head_hidden_channels: int,
            output_support_size: int,
            flatten_input_size_for_value_head: int,
            flatten_input_size_for_policy_head: int,
            downsample: bool = False,
            last_linear_layer_init_zero: bool = True,
            activation: nn.Module = nn.ReLU(inplace=True),
            norm_type: Optional[str] = 'BN',
    ) -> None:
        """
        Overview:
            The definition of policy and value prediction network, which is used to predict value and policy by the
            given latent state.
        Arguments:
            - observation_shape (:obj:`SequenceType`): The shape of observation space, e.g. (C, H, W) for image.
            - action_space_size: (:obj:`int`): Action space size, usually an integer number for discrete action space.
            - num_res_blocks (:obj:`int`): The number of res blocks in AlphaZero model.
            - num_channels (:obj:`int`): The channels of hidden states.
            - value_head_channels (:obj:`int`): The channels of value head.
            - policy_head_channels (:obj:`int`): The channels of policy head.
            - value_head_hidden_channels (:obj:`SequenceType`): The number of hidden layers used in value head (MLP head).
            - policy_head_hidden_channels (:obj:`SequenceType`): The number of hidden layers used in policy head (MLP head).
            - output_support_size (:obj:`int`): The size of categorical value output.
            - self_supervised_learning_loss (:obj:`bool`): Whether to use self_supervised_learning related networks \
            - flatten_input_size_for_value_head (:obj:`int`): The size of flatten hidden states, i.e. the input size \
                of the value head.
            - flatten_input_size_for_policy_head (:obj:`int`): The size of flatten hidden states, i.e. the input size \
                of the policy head.
            - downsample (:obj:`bool`): Whether to do downsampling for observations in ``representation_network``.
            - last_linear_layer_init_zero (:obj:`bool`): Whether to use zero initializations for the last layer of \
                dynamics/prediction mlp, default sets it to True.
            - activation (:obj:`Optional[nn.Module]`): Activation function used in network, which often use in-place \
                operation to speedup, e.g. ReLU(inplace=True).
            - norm_type (:obj:`str`): The type of normalization in networks. defaults to 'BN'.
        """
        super(PredictionNetwork, self).__init__()
        assert norm_type in ['BN', 'LN'], "norm_type must in ['BN', 'LN']"

        self.resblocks = nn.ModuleList(
            [
                ResBlock(
                    in_channels=num_channels, activation=activation, norm_type=norm_type, res_type='basic', bias=False
                ) for _ in range(num_res_blocks)
            ]
        )

        self.conv1x1_value = nn.Conv2d(num_channels, value_head_channels, 1)
        self.conv1x1_policy = nn.Conv2d(num_channels, policy_head_channels, 1)

        if observation_shape[1] == 96:
            latent_shape = (observation_shape[1] / 16, observation_shape[2] / 16)
        elif observation_shape[1] == 64:
            latent_shape = (observation_shape[1] / 8, observation_shape[2] / 8)

        if norm_type == 'BN':
            self.norm_value = nn.BatchNorm2d(value_head_channels)
            self.norm_policy = nn.BatchNorm2d(policy_head_channels)
        elif norm_type == 'LN':
            if downsample:
                self.norm_value = nn.LayerNorm(
                    [value_head_channels, *latent_shape],
                    eps=1e-5)
                self.norm_policy = nn.LayerNorm([policy_head_channels, *latent_shape], eps=1e-5)
            else:
                self.norm_value = nn.LayerNorm([value_head_channels, observation_shape[-2], observation_shape[-1]],
                                               eps=1e-5)
                self.norm_policy = nn.LayerNorm([policy_head_channels, observation_shape[-2], observation_shape[-1]],
                                                eps=1e-5)

        self.flatten_input_size_for_value_head = flatten_input_size_for_value_head
        self.flatten_input_size_for_policy_head = flatten_input_size_for_policy_head

        self.activation = activation

        self.fc_value = MLP_V2(
            in_channels=self.flatten_input_size_for_value_head,
            hidden_channels=value_head_hidden_channels,
            out_channels=output_support_size,
            activation=self.activation,
            norm_type=norm_type,
            output_activation=False,
            output_norm=False,
            # last_linear_layer_init_zero=True is beneficial for convergence speed.
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )
        self.fc_policy = MLP_V2(
            in_channels=self.flatten_input_size_for_policy_head,
            hidden_channels=policy_head_hidden_channels,
            out_channels=action_space_size,
            activation=self.activation,
            norm_type=norm_type,
            output_activation=False,
            output_norm=False,
            # last_linear_layer_init_zero=True is beneficial for convergence speed.
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )

    def forward(self, latent_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Overview:
            Forward computation of the prediction network.
        Arguments:
            - latent_state (:obj:`torch.Tensor`): input tensor with shape (B, latent_state_dim).
        Returns:
            - policy (:obj:`torch.Tensor`): policy tensor with shape (B, action_space_size).
            - value (:obj:`torch.Tensor`): value tensor with shape (B, output_support_size).
        """
        for res_block in self.resblocks:
            latent_state = res_block(latent_state)

        value = self.conv1x1_value(latent_state)
        value = self.norm_value(value)
        value = self.activation(value)

        policy = self.conv1x1_policy(latent_state)
        policy = self.norm_policy(policy)
        policy = self.activation(policy)

        value = value.reshape(-1, self.flatten_input_size_for_value_head)
        policy = policy.reshape(-1, self.flatten_input_size_for_policy_head)

        value = self.fc_value(value)
        policy = self.fc_policy(policy)
        return policy, value


class PredictionNetworkMLP(nn.Module):

    def __init__(
            self,
            action_space_size,
            num_channels,
            common_layer_num: int = 2,
            value_head_hidden_channels: SequenceType = [32],
            policy_head_hidden_channels: SequenceType = [32],
            output_support_size: int = 601,
            last_linear_layer_init_zero: bool = True,
            activation: Optional[nn.Module] = nn.ReLU(inplace=True),
            norm_type: Optional[str] = 'BN',
    ):
        """
        Overview:
            The definition of policy and value prediction network with Multi-Layer Perceptron (MLP),
            which is used to predict value and policy by the given latent state.
        Arguments:
            - action_space_size: (:obj:`int`): Action space size, usually an integer number. For discrete action \
                space, it is the number of discrete actions.
            - num_channels (:obj:`int`): The channels of latent states.
            - value_head_hidden_channels (:obj:`SequenceType`): The number of hidden layers used in value head (MLP head).
            - policy_head_hidden_channels (:obj:`SequenceType`): The number of hidden layers used in policy head (MLP head).
            - output_support_size (:obj:`int`): The size of categorical value output.
            - last_linear_layer_init_zero (:obj:`bool`): Whether to use zero initializations for the last layer of \
                dynamics/prediction mlp, default sets it to True.
            - activation (:obj:`Optional[nn.Module]`): Activation function used in network, which often use in-place \
                operation to speedup, e.g. ReLU(inplace=True).
            - norm_type (:obj:`str`): The type of normalization in networks. defaults to 'BN'.
        """
        super().__init__()
        self.num_channels = num_channels

        # ******* common backbone ******
        self.fc_prediction_common = MLP(
            in_channels=self.num_channels,
            hidden_channels=self.num_channels,
            out_channels=self.num_channels,
            layer_num=common_layer_num,
            activation=activation,
            norm_type=norm_type,
            output_activation=True,
            output_norm=True,
            # last_linear_layer_init_zero=False is important for convergence
            last_linear_layer_init_zero=False,
        )

        # ******* value and policy head ******
        self.fc_value_head = MLP_V2(
            in_channels=self.num_channels,
            hidden_channels=value_head_hidden_channels,
            out_channels=output_support_size,
            activation=activation,
            norm_type=norm_type,
            output_activation=False,
            output_norm=False,
            # last_linear_layer_init_zero=True is beneficial for convergence speed.
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )
        self.fc_policy_head = MLP_V2(
            in_channels=self.num_channels,
            hidden_channels=policy_head_hidden_channels,
            out_channels=action_space_size,
            activation=activation,
            norm_type=norm_type,
            output_activation=False,
            output_norm=False,
            # last_linear_layer_init_zero=True is beneficial for convergence speed.
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )

    def forward(self, latent_state: torch.Tensor):
        """
        Overview:
            Forward computation of the prediction network.
        Arguments:
            - latent_state (:obj:`torch.Tensor`): input tensor with shape (B, latent_state_dim).
        Returns:
            - policy (:obj:`torch.Tensor`): policy tensor with shape (B, action_space_size).
            - value (:obj:`torch.Tensor`): value tensor with shape (B, output_support_size).
        """
        x_prediction_common = self.fc_prediction_common(latent_state)

        value = self.fc_value_head(x_prediction_common)
        policy = self.fc_policy_head(x_prediction_common)
        return policy, value


class PredictionHiddenNetwork(nn.Module):

    def __init__(
            self,
            observation_shape: SequenceType,
            action_space_size: int,
            num_res_blocks: int,
            num_channels: int,
            value_head_channels: int,
            policy_head_channels: int,
            value_head_hidden_channels: int,
            policy_head_hidden_channels: int,
            output_support_size: int,
            flatten_input_size_for_value_head: int,
            flatten_input_size_for_policy_head: int,
            downsample: bool = False,
            last_linear_layer_init_zero: bool = True,
            activation: nn.Module = nn.ReLU(inplace=True),
            norm_type: Optional[str] = 'BN',
            gru_hidden_size: int = 512,
    ) -> None:
        """
        Overview:
            The definition of policy and value prediction network, which is used to predict value and policy by the
            given latent state.
        Arguments:
            - observation_shape (:obj:`SequenceType`): The shape of observation space, e.g. (C, H, W) for image.
            - action_space_size: (:obj:`int`): Action space size, usually an integer number for discrete action space.
            - num_res_blocks (:obj:`int`): The number of res blocks in AlphaZero model.
            - num_channels (:obj:`int`): The channels of hidden states.
            - value_head_channels (:obj:`int`): The channels of value head.
            - policy_head_channels (:obj:`int`): The channels of policy head.
            - value_head_hidden_channels (:obj:`SequenceType`): The number of hidden layers used in value head (MLP head).
            - policy_head_hidden_channels (:obj:`SequenceType`): The number of hidden layers used in policy head (MLP head).
            - output_support_size (:obj:`int`): The size of categorical value output.
            - self_supervised_learning_loss (:obj:`bool`): Whether to use self_supervised_learning related networks \
            - flatten_input_size_for_value_head (:obj:`int`): The size of flatten hidden states, i.e. the input size \
                of the value head.
            - flatten_input_size_for_policy_head (:obj:`int`): The size of flatten hidden states, i.e. the input size \
                of the policy head.
            - downsample (:obj:`bool`): Whether to do downsampling for observations in ``representation_network``.
            - last_linear_layer_init_zero (:obj:`bool`): Whether to use zero initializations for the last layer of \
                dynamics/prediction mlp, default sets it to True.
            - activation (:obj:`Optional[nn.Module]`): Activation function used in network, which often use in-place \
                operation to speedup, e.g. ReLU(inplace=True).
            - norm_type (:obj:`str`): The type of normalization in networks. defaults to 'BN'.
        """
        super(PredictionHiddenNetwork, self).__init__()
        assert norm_type in ['BN', 'LN'], "norm_type must in ['BN', 'LN']"

        self.observation_shape = observation_shape
        self.gru_hidden_size = gru_hidden_size
        self.resblocks = nn.ModuleList(
            [
                ResBlock(
                    in_channels=num_channels, activation=activation, norm_type=norm_type, res_type='basic', bias=False
                ) for _ in range(num_res_blocks)
            ]
        )

        self.conv1x1_value = nn.Conv2d(num_channels, value_head_channels, 1)
        self.conv1x1_policy = nn.Conv2d(num_channels, policy_head_channels, 1)

        if norm_type == 'BN':
            self.norm_value = nn.BatchNorm2d(value_head_channels)
            self.norm_policy = nn.BatchNorm2d(policy_head_channels)
        elif norm_type == 'LN':
            if downsample:
                self.norm_value = nn.LayerNorm(
                    [value_head_channels, math.ceil(observation_shape[-2] / 16), math.ceil(observation_shape[-1] / 16)],
                    eps=1e-5)
                self.norm_policy = nn.LayerNorm([policy_head_channels, math.ceil(observation_shape[-2] / 16),
                                                 math.ceil(observation_shape[-1] / 16)], eps=1e-5)
            else:
                self.norm_value = nn.LayerNorm([value_head_channels, observation_shape[-2], observation_shape[-1]],
                                               eps=1e-5)
                self.norm_policy = nn.LayerNorm([policy_head_channels, observation_shape[-2], observation_shape[-1]],
                                                eps=1e-5)

        self.flatten_input_size_for_value_head = flatten_input_size_for_value_head
        self.flatten_input_size_for_policy_head = flatten_input_size_for_policy_head

        self.activation = activation

        self.fc_value = MLP(
            in_channels=self.flatten_input_size_for_value_head + self.gru_hidden_size,
            hidden_channels=value_head_hidden_channels[0],
            out_channels=output_support_size,
            layer_num=len(value_head_hidden_channels) + 1,
            activation=self.activation,
            norm_type=norm_type,
            output_activation=False,
            output_norm=False,
            # last_linear_layer_init_zero=True is beneficial for convergence speed.
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )
        self.fc_policy = MLP(
            in_channels=self.flatten_input_size_for_policy_head + self.gru_hidden_size,
            hidden_channels=policy_head_hidden_channels[0],
            out_channels=action_space_size,
            layer_num=len(policy_head_hidden_channels) + 1,
            activation=self.activation,
            norm_type=norm_type,
            output_activation=False,
            output_norm=False,
            # last_linear_layer_init_zero=True is beneficial for convergence speed.
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )

    def forward(self, latent_state: torch.Tensor, world_model_latent_history: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """
        Overview:
            Forward computation of the prediction network.
        Arguments:
            - latent_state (:obj:`torch.Tensor`): input tensor with shape (B, latent_state_dim).
        Returns:
            - policy (:obj:`torch.Tensor`): policy tensor with shape (B, action_space_size).
            - value (:obj:`torch.Tensor`): value tensor with shape (B, output_support_size).
        """
        for res_block in self.resblocks:
            latent_state = res_block(latent_state)

        value = self.conv1x1_value(latent_state)
        value = self.norm_value(value)
        value = self.activation(value)

        policy = self.conv1x1_policy(latent_state)
        policy = self.norm_policy(policy)
        policy = self.activation(policy)

        latent_state_value = value.reshape(-1, self.flatten_input_size_for_value_head)
        latent_state_policy = policy.reshape(-1, self.flatten_input_size_for_policy_head)

        # TODO: world_model_latent_history.squeeze(0) shape: (num_layers * num_directions, batch_size, hidden_size) ->  ( batch_size, hidden_size)
        latent_history_value = torch.cat([latent_state_value, world_model_latent_history.squeeze(0)], dim=1)
        latent_history_policy = torch.cat([latent_state_policy, world_model_latent_history.squeeze(0)], dim=1)

        value = self.fc_value(latent_history_value)
        policy = self.fc_policy(latent_history_policy)
        return policy, value

def get_act_fn(act_fn):
    if act_fn == 'relu':
        return nn.ReLU()
    elif act_fn == 'leaky_relu':
        return nn.LeakyReLU()
    elif act_fn == 'elu':
        return nn.ELU()
    elif act_fn == 'sigmoid':
        return nn.Sigmoid()
    elif act_fn == 'softplus':
        return nn.Softplus()
    else:
        raise ValueError('Invalid argument for `act_fn`.')

def unsorted_segment_sum(tensor, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    result_shape = (num_segments, tensor.size(1))
    result = tensor.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, tensor.size(1))
    result.scatter_add_(0, segment_ids, tensor)
    return result

def to_one_hot(indices, max_index):
    """Get one-hot encoding of index tensors."""
    zeros = torch.zeros(
        indices.size()[0], max_index, dtype=torch.float32,
        device=indices.device)
    return zeros.scatter_(1, indices.unsqueeze(1), 1)

def make_node_mlp_layers(num_layers, input_dim, hidden_dim, output_dim, act_fn, layer_norm):
    layers = []

    for idx in range(num_layers):

        if idx == 0:
            # first layer, input_dim => hidden_dim
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(get_act_fn(act_fn))
        elif idx == num_layers - 2:
            # layer before the last, add layer norm
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(get_act_fn(act_fn))
        elif idx == num_layers - 1:
            # last layer, hidden_dim => output_dim and no activation
            layers.append(nn.Linear(hidden_dim, output_dim))
        else:
            # all other layers, hidden_dim => hidden_dim
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(get_act_fn(act_fn))

    return layers

class GNN(nn.Module):

    def __init__(self, input_dim, hidden_dim, action_dim, num_objects, ignore_action=False, copy_action=False,
                 act_fn='relu', layer_norm=True, num_layers=3, use_interactions=True, edge_actions=False,
                 output_dim=None):
        super(GNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        if self.output_dim is None:
            self.output_dim = self.input_dim

        self.num_objects = num_objects
        self.ignore_action = ignore_action
        self.copy_action = copy_action
        self.use_interactions = use_interactions
        self.edge_actions = edge_actions
        self.num_layers = num_layers

        if self.ignore_action:
            self.action_dim = 0
        else:
            self.action_dim = action_dim

        tmp_action_dim = self.action_dim
        edge_mlp_input_size = self.input_dim * 2 + int(self.edge_actions) * tmp_action_dim

        self.edge_mlp = nn.Sequential(*self.make_node_mlp_layers_(
            edge_mlp_input_size, self.hidden_dim, act_fn, layer_norm
        ))

        if self.num_objects == 1 or not self.use_interactions:
            node_input_dim = self.input_dim + tmp_action_dim
        else:
            node_input_dim = hidden_dim + self.input_dim + tmp_action_dim

        self.node_mlp = nn.Sequential(*self.make_node_mlp_layers_(
            node_input_dim, self.output_dim, act_fn, layer_norm
        ))

        self.edge_list = None
        self.batch_size = 0

    def _edge_model(self, source, target, action=None):
        if action is None:
            x = [source, target]
        else:
            x = [source, target, action]

        out = torch.cat(x, dim=1)
        return self.edge_mlp(out)

    def _node_model(self, node_attr, edge_index, edge_attr):
        if edge_attr is not None:
            row, col = edge_index
            agg = unsorted_segment_sum(
                edge_attr, row, num_segments=node_attr.size(0))
            out = torch.cat([node_attr, agg], dim=1)
        else:
            out = node_attr
        return self.node_mlp(out)

    def _get_edge_list_fully_connected(self, batch_size, num_objects, device):
        # Only re-evaluate if necessary (e.g. if batch size changed).
        if self.edge_list is None or self.batch_size != batch_size:
            self.batch_size = batch_size

            # Create fully-connected adjacency matrix for single sample.
            adj_full = torch.ones(num_objects, num_objects)

            # Remove diagonal.
            adj_full -= torch.eye(num_objects)
            self.edge_list = adj_full.nonzero()

            # Copy `batch_size` times and add offset.
            self.edge_list = self.edge_list.repeat(batch_size, 1)
            offset = torch.arange(
                0, batch_size * num_objects, num_objects).unsqueeze(-1)
            offset = offset.expand(batch_size, num_objects * (num_objects - 1))
            offset = offset.contiguous().view(-1)
            self.edge_list += offset.unsqueeze(-1)

            # Transpose to COO format -> Shape: [2, num_edges].
            self.edge_list = self.edge_list.transpose(0, 1)
            self.edge_list = self.edge_list.to(device)

        return self.edge_list

    def process_action_(self, action):
        if self.copy_action:
            if action.shape[1] == 1 and (action.dtype in (torch.int32, torch.int64)):
                # action is an integer
                action = action.squeeze(1)
                action_vec = to_one_hot(action, self.action_dim).repeat(1, self.num_objects)
            else:
                # action is a vector
                action_vec = action.repeat(1, self.num_objects)
            # mix node and batch dimension
            action_vec = action_vec.reshape(-1, self.action_dim).float()
        else:
            # we have a separate action for each node
            if action.shape[1] == 1 and (action.dtype in (torch.int32, torch.int64)):
                # index for both object and action
                action = action.squeeze(1)
                action_vec = to_one_hot(action, self.action_dim * self.num_objects)
                action_vec = action_vec.reshape(-1, self.action_dim)
            else:
                action_vec = action.reshape(action.size(0), self.action_dim * self.num_objects)
                action_vec = action_vec.reshape(-1, self.action_dim)

        return action_vec

    def forward(self, states, action):

        device = states.device
        batch_size = states.size(0)
        num_nodes = states.size(1)

        # states: [batch_size (B), num_objects, embedding_dim]
        # node_attr: Flatten states tensor to [B * num_objects, embedding_dim]
        node_attr = states.reshape(-1, self.input_dim)

        action_vec = None
        if not self.ignore_action:
            action_vec = self.process_action_(action)

        edge_attr = None
        edge_index = None

        if num_nodes > 1 and self.use_interactions:
            # edge_index: [B * (num_objects*[num_objects-1]), 2] edge list
            edge_index = self._get_edge_list_fully_connected(
                batch_size, num_nodes, device)

            row, col = edge_index
            edge_attr = self._edge_model(node_attr[row], node_attr[col], action_vec[row] if self.edge_actions else None)

        if not self.ignore_action:
            # Attach action to each state
            node_attr = torch.cat([node_attr, action_vec], dim=-1)

        node_attr = self._node_model(
            node_attr, edge_index, edge_attr)

        # [batch_size, num_nodes, hidden_dim]
        node_attr = node_attr.view(batch_size, num_nodes, -1)

        return node_attr

    def make_node_mlp_layers_(self, input_dim, output_dim, act_fn, layer_norm):
        return make_node_mlp_layers(self.num_layers, input_dim, self.hidden_dim, output_dim, act_fn, layer_norm)

class OCDynamicsNetwork(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_encoding_dim: int = 2,
        num_res_blocks: int = 1,
        num_channels: int = 64,
        reward_head_channels: int = 64,
        reward_head_hidden_channels: Sequence[int] = [32],
        output_support_size: int = 601,
        flatten_input_size_for_reward_head: int = 64,
        downsample: bool = False,
        rnn_hidden_size: int = 512,
        last_linear_layer_init_zero: bool = True,
        activation: Optional[nn.Module] = nn.ReLU(inplace=True),
        norm_type: Optional[str] = 'BN',
        embedding_dim: int = 256,
        group_size: int = 8,
        use_sim_norm: bool = False,
        res_connection_in_dynamics: bool = True,
    ):
        """
        Define the Dynamics Network for predicting the next latent state, reward,
        and reward hidden state based on the current state and action.

        Arguments:
            observation_shape (Sequence[int]): Shape of the input observation, e.g., (12, 96, 96).
            action_encoding_dim (int): Dimension of the action encoding.
            num_res_blocks (int): Number of residual blocks in the MuZero model.
            num_channels (int): Number of channels in the latent state.
            reward_head_channels (int): Number of channels in the reward head.
            reward_head_hidden_channels (Sequence[int]): Hidden layers in the reward head MLP.
            output_support_size (int): Size of the output for reward classification.
            flatten_input_size_for_reward_head (int): Flattened output size for the reward head.
            downsample (bool): Whether to downsample the input observation. Default is False.
            rnn_hidden_size (int): Hidden size of the LSTM in the dynamics network.
            last_linear_layer_init_zero (bool): Whether to initialize the last reward MLP layer to zero. Default is True.
            activation (Optional[nn.Module]): Activation function used in the network. Default is ReLU(inplace=True).
            norm_type (Optional[str]): Type of normalization used in the network. Default is 'BN'.
            embedding_dim (int): Embedding dimension if using SimNorm.
            group_size (int): Group size for SimNorm.
            use_sim_norm (bool): Whether to use SimNorm. Default is False.
            res_connection_in_dynamics (bool): Whether to use residual connections in the dynamics. Default is True.
        """
        super().__init__()
        assert norm_type in ['BN', 'LN'], "Normalization type must be 'BN' or 'LN'"
        assert num_channels > action_encoding_dim, f'Number of channels:{num_channels} <= action encoding dimension:{action_encoding_dim}'

        self.action_encoding_dim = action_encoding_dim
        self.num_channels = num_channels
        self.rnn_hidden_size = rnn_hidden_size
        self.flatten_input_size_for_reward_head = flatten_input_size_for_reward_head

        self.num_channels_of_latent_state = num_channels - self.action_encoding_dim
        self.activation = activation

        # 1x1 convolution to adjust the number of channels
        self.conv = nn.Conv2d(num_channels, self.num_channels_of_latent_state, kernel_size=1, stride=1, bias=False)

        # Choose between BatchNorm or LayerNorm
        if norm_type == 'BN':
            self.norm_common = nn.BatchNorm2d(self.num_channels_of_latent_state)
        elif norm_type == 'LN':
            self.norm_common = nn.LayerNorm(
                [self.num_channels_of_latent_state,
                 math.ceil(observation_shape[-2] / (16 if downsample else 1)),
                 math.ceil(observation_shape[-1] / (16 if downsample else 1))]
            )

        # Residual Blocks
        self.resblocks = nn.ModuleList([
            ResBlock(
                in_channels=self.num_channels_of_latent_state,
                activation=self.activation,
                norm_type=norm_type,
                res_type='basic',
                bias=False
            ) for _ in range(num_res_blocks)
        ])

        # Reward prediction residual blocks
        self.reward_resblocks = nn.ModuleList([
            ResBlock(
                in_channels=self.num_channels_of_latent_state,
                activation=self.activation,
                norm_type=norm_type,
                res_type='basic',
                bias=False
            ) for _ in range(num_res_blocks)
        ])

        # 1x1 convolution to adjust the number of reward head channels
        self.conv1x1_reward = nn.Conv2d(self.num_channels_of_latent_state, reward_head_channels, 1)

        # Choose normalization before LSTM
        if norm_type == 'BN':
            self.norm_before_lstm = nn.BatchNorm2d(reward_head_channels)
        elif norm_type == 'LN':
            self.norm_before_lstm = nn.LayerNorm(
                [reward_head_channels,
                 math.ceil(observation_shape[-2] / (16 if downsample else 1)),
                 math.ceil(observation_shape[-1] / (16 if downsample else 1))]
            )

        # Reward head MLP
        self.fc_reward_head = MLP(
            self.rnn_hidden_size,
            hidden_channels=reward_head_hidden_channels[0],
            layer_num=len(reward_head_hidden_channels) + 1,
            out_channels=output_support_size,
            activation=activation,
            norm_type=norm_type,
            output_activation=False,
            output_norm=False,
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )

        self.latent_state_dim = self.flatten_input_size_for_reward_head

        self.gru = nn.GRU(input_size=self.latent_state_dim, hidden_size=self.rnn_hidden_size, num_layers=1, batch_first=True)

        # Compute output dimensions and shapes based on whether downsampling is used
        if downsample:
            ceil_size = math.ceil(observation_shape[1] / 16) * math.ceil(observation_shape[2] / 16)
            self.output_dim = self.num_channels_of_latent_state * ceil_size
            self.output_shape = (
                self.num_channels_of_latent_state,
                math.ceil(observation_shape[1] / 16),
                math.ceil(observation_shape[2] / 16)
            )
        else:
            self.output_dim = self.num_channels_of_latent_state * observation_shape[1] * observation_shape[2]
            self.output_shape = (self.num_channels_of_latent_state, observation_shape[1], observation_shape[2])

        # Flatten dimension of the latent state
        self.latent_state_flatten_dim = 64 * 8 * 8

        # Linear layer to adjust dimensions
        self.linear_common = nn.Linear(self.latent_state_flatten_dim, self.latent_state_dim, bias=False)

        # Dynamics head MLP
        self.fc_dynamics_head = MLP(
            self.rnn_hidden_size,
            hidden_channels=self.rnn_hidden_size,
            layer_num=2,
            out_channels=self.latent_state_flatten_dim,
            activation=activation,
            norm_type=norm_type,
            output_activation=True,
            output_norm=True,
            last_linear_layer_init_zero=False  # Important for convergence
        )

        self.res_connection_in_dynamics = res_connection_in_dynamics
        self.use_sim_norm = use_sim_norm

        # If using SimNorm
        if self.use_sim_norm:
            self.embedding_dim = embedding_dim
            self.last_linear = nn.Linear(self.latent_state_flatten_dim, self.embedding_dim, bias=False)
            init.kaiming_normal_(self.last_linear.weight, mode='fan_out', nonlinearity='relu')
            self.sim_norm = SimNorm(simnorm_dim=group_size)

    def forward(
        self,
        state_action_encoding: torch.Tensor,
        dynamics_hidden_state: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Forward pass for the Dynamics Network. Predict the next latent state,
        the next dynamics hidden state, and the reward based on the current state-action encoding.

        Arguments:
            state_action_encoding (torch.Tensor): State-action encoding, a concatenation of the latent state and action encoding.
            dynamics_hidden_state (Tuple[torch.Tensor, torch.Tensor]): Hidden state for the LSTM related to reward.

        Returns:
            next_latent_state (torch.Tensor): Next latent state.
            next_dynamics_hidden_state (Tuple[torch.Tensor, torch.Tensor]): Next hidden state for the LSTM.
            reward (torch.Tensor): Predicted reward.
        """
        # Extract latent state
        latent_state = state_action_encoding[:, :-self.action_encoding_dim, :, :]

        # Adjust channels and normalize
        x = self.conv(state_action_encoding)
        x = self.norm_common(x)
        x += latent_state  # Residual connection
        x = self.activation(x)

        # Pass through residual blocks
        for block in self.resblocks:
            x = block(x)

        # Reshape and transform
        x = self.linear_common(x.view(-1, self.latent_state_flatten_dim))
        x = self.activation(x).unsqueeze(1)

        # ==================== GRU backbone ==================
        # Pass through GRU
        gru_outputs, next_dynamics_hidden_state = self.gru(x, dynamics_hidden_state)

        # Predict reward
        reward = self.fc_reward_head(gru_outputs.squeeze(1))

        # Predict next latent state
        next_latent_state_encoding = self.fc_dynamics_head(gru_outputs.squeeze(1))

        # Residual connection
        if self.res_connection_in_dynamics:
            next_latent_state = next_latent_state_encoding.view(latent_state.shape) + latent_state
        else:
            next_latent_state = next_latent_state_encoding.view(latent_state.shape)

        # Apply SimNorm if used
        if self.use_sim_norm:
            next_latent_state = self.sim_norm(next_latent_state)

        return next_latent_state, next_dynamics_hidden_state, reward


class OCPredictionNetwork(nn.Module):

    def __init__(
            self,
            observation_shape: SequenceType,
            action_space_size: int,
            num_res_blocks: int,
            num_channels: int,
            value_head_channels: int,
            policy_head_channels: int,
            value_head_hidden_channels: int,
            policy_head_hidden_channels: int,
            output_support_size: int,
            flatten_input_size_for_value_head: int,
            flatten_input_size_for_policy_head: int,
            downsample: bool = False,
            last_linear_layer_init_zero: bool = True,
            activation: nn.Module = nn.ReLU(inplace=True),
            norm_type: Optional[str] = 'BN',
    ) -> None:
        """
        Overview:
            The definition of policy and value prediction network, which is used to predict value and policy by the
            given latent state.
        Arguments:
            - observation_shape (:obj:`SequenceType`): The shape of observation space, e.g. (C, H, W) for image.
            - action_space_size: (:obj:`int`): Action space size, usually an integer number for discrete action space.
            - num_res_blocks (:obj:`int`): The number of res blocks in AlphaZero model.
            - num_channels (:obj:`int`): The channels of hidden states.
            - value_head_channels (:obj:`int`): The channels of value head.
            - policy_head_channels (:obj:`int`): The channels of policy head.
            - value_head_hidden_channels (:obj:`SequenceType`): The number of hidden layers used in value head (MLP head).
            - policy_head_hidden_channels (:obj:`SequenceType`): The number of hidden layers used in policy head (MLP head).
            - output_support_size (:obj:`int`): The size of categorical value output.
            - self_supervised_learning_loss (:obj:`bool`): Whether to use self_supervised_learning related networks \
            - flatten_input_size_for_value_head (:obj:`int`): The size of flatten hidden states, i.e. the input size \
                of the value head.
            - flatten_input_size_for_policy_head (:obj:`int`): The size of flatten hidden states, i.e. the input size \
                of the policy head.
            - downsample (:obj:`bool`): Whether to do downsampling for observations in ``representation_network``.
            - last_linear_layer_init_zero (:obj:`bool`): Whether to use zero initializations for the last layer of \
                dynamics/prediction mlp, default sets it to True.
            - activation (:obj:`Optional[nn.Module]`): Activation function used in network, which often use in-place \
                operation to speedup, e.g. ReLU(inplace=True).
            - norm_type (:obj:`str`): The type of normalization in networks. defaults to 'BN'.
        """
        super(PredictionNetwork, self).__init__()
        assert norm_type in ['BN', 'LN'], "norm_type must in ['BN', 'LN']"

        self.resblocks = nn.ModuleList(
            [
                ResBlock(
                    in_channels=num_channels, activation=activation, norm_type=norm_type, res_type='basic', bias=False
                ) for _ in range(num_res_blocks)
            ]
        )

        self.conv1x1_value = nn.Conv2d(num_channels, value_head_channels, 1)
        self.conv1x1_policy = nn.Conv2d(num_channels, policy_head_channels, 1)

        if observation_shape[1] == 96:
            latent_shape = (observation_shape[1] / 16, observation_shape[2] / 16)
        elif observation_shape[1] == 64:
            latent_shape = (observation_shape[1] / 8, observation_shape[2] / 8)

        if norm_type == 'BN':
            self.norm_value = nn.BatchNorm2d(value_head_channels)
            self.norm_policy = nn.BatchNorm2d(policy_head_channels)
        elif norm_type == 'LN':
            if downsample:
                self.norm_value = nn.LayerNorm(
                    [value_head_channels, *latent_shape],
                    eps=1e-5)
                self.norm_policy = nn.LayerNorm([policy_head_channels, *latent_shape], eps=1e-5)
            else:
                self.norm_value = nn.LayerNorm([value_head_channels, observation_shape[-2], observation_shape[-1]],
                                               eps=1e-5)
                self.norm_policy = nn.LayerNorm([policy_head_channels, observation_shape[-2], observation_shape[-1]],
                                                eps=1e-5)

        self.flatten_input_size_for_value_head = flatten_input_size_for_value_head
        self.flatten_input_size_for_policy_head = flatten_input_size_for_policy_head

        self.activation = activation

        self.fc_value = MLP_V2(
            in_channels=self.flatten_input_size_for_value_head,
            hidden_channels=value_head_hidden_channels,
            out_channels=output_support_size,
            activation=self.activation,
            norm_type=norm_type,
            output_activation=False,
            output_norm=False,
            # last_linear_layer_init_zero=True is beneficial for convergence speed.
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )
        self.fc_policy = MLP_V2(
            in_channels=self.flatten_input_size_for_policy_head,
            hidden_channels=policy_head_hidden_channels,
            out_channels=action_space_size,
            activation=self.activation,
            norm_type=norm_type,
            output_activation=False,
            output_norm=False,
            # last_linear_layer_init_zero=True is beneficial for convergence speed.
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )

    def forward(self, latent_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Overview:
            Forward computation of the prediction network.
        Arguments:
            - latent_state (:obj:`torch.Tensor`): input tensor with shape (B, latent_state_dim).
        Returns:
            - policy (:obj:`torch.Tensor`): policy tensor with shape (B, action_space_size).
            - value (:obj:`torch.Tensor`): value tensor with shape (B, output_support_size).
        """
        for res_block in self.resblocks:
            latent_state = res_block(latent_state)

        value = self.conv1x1_value(latent_state)
        value = self.norm_value(value)
        value = self.activation(value)

        policy = self.conv1x1_policy(latent_state)
        policy = self.norm_policy(policy)
        policy = self.activation(policy)

        value = value.reshape(-1, self.flatten_input_size_for_value_head)
        policy = policy.reshape(-1, self.flatten_input_size_for_policy_head)

        value = self.fc_value(value)
        policy = self.fc_policy(policy)
        return policy, value