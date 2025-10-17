import copy
import math
from typing import Optional, Tuple, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from ding.torch_utils import MLP, ResBlock
from ding.utils import MODEL_REGISTRY, SequenceType

#from lzero.model.common import SimNorm
from .common import MZRNNNetworkOutput
from .utils import renormalize


@MODEL_REGISTRY.register('ObjectZeroRNNFullObsModel')
class ObjectZeroRNNFullObsModel(nn.Module):

    def __init__(
            self,
            observation_shape: SequenceType = (6, 64),
            action_space_size: int = 6,
            rnn_hidden_size: int = 512,
            num_res_blocks: int = 1,
            num_channels: int = 64,
            reward_head_channels: int = 16,
            value_head_channels: int = 16,
            policy_head_channels: int = 16,
            reward_head_hidden_channels: SequenceType = [32],
            value_head_hidden_channels: SequenceType = [32],
            policy_head_hidden_channels: SequenceType = [32],
            reward_support_range: SequenceType =(-300., 301., 1.),
            value_support_range: SequenceType =(-300., 301., 1.),
            proj_hid: int = 1024,
            proj_out: int = 1024,
            pred_hid: int = 512,
            pred_out: int = 1024,
            self_supervised_learning_loss: bool = True,
            categorical_distribution: bool = True,
            last_linear_layer_init_zero: bool = True,
            state_norm: bool = False,
            downsample: bool = False,
            activation: Optional[nn.Module] = nn.ReLU(inplace=True),
            norm_type: Optional[str] = 'BN',
            discrete_action_encoding_type: str = 'one_hot',
            context_length_init: int = 5,
            use_sim_norm: bool = False,
            analysis_sim_norm: bool = False,
            collector_env_num: int = 8,
            *args,
            **kwargs
    ) -> None:
        """
        Overview:
            The definition of the network model for MuZeroRNNFullObs, a variant of MuZero, involves the use of a recurrent neural network to predict both reward/next_latent_state and value/policy.
            This model fully utilizes observation information and retains training settings similar to UniZero but employs a GRU backbone.
            During the inference phase, the hidden state of the GRU is reset and cleared every H_infer steps.
            This variant is proposed in the UniZero paper: https://arxiv.org/abs/2406.10667.
        Arguments:
            - observation_shape (:obj:`SequenceType`): Observation space shape, e.g. [C, W, H]=[12, 96, 96] for Atari.
            - action_space_size: (:obj:`int`): Action space size, usually an integer number for discrete action space.
            - rnn_hidden_size (:obj:`int`): The hidden size of LSTM in dynamics network to predict reward.
            - num_res_blocks (:obj:`int`): The number of res blocks in MuZero model.
            - num_channels (:obj:`int`): The channels of hidden states.
            - reward_head_channels (:obj:`int`): The channels of reward head.
            - value_head_channels (:obj:`int`): The channels of value head.
            - policy_head_channels (:obj:`int`): The channels of policy head.
            - reward_head_hidden_channels (:obj:`SequenceType`): The number of hidden layers of the reward head (MLP head).
            - value_head_hidden_channels (:obj:`SequenceType`): The number of hidden layers used in value head (MLP head).
            - policy_head_hidden_channels (:obj:`SequenceType`): The number of hidden layers used in policy head (MLP head).
            - reward_support_range (:obj:`SequenceType`): The range of categorical reward output
            - value_support_range (:obj:`SequenceType`): The range of categorical value output.
            - proj_hid (:obj:`int`): The size of projection hidden layer.
            - proj_out (:obj:`int`): The size of projection output layer.
            - pred_hid (:obj:`int`): The size of prediction hidden layer.
            - pred_out (:obj:`int`): The size of prediction output layer.
            - categorical_distribution (:obj:`bool`): Whether to use discrete support to represent categorical \
                distribution for value and reward/reward.
            - last_linear_layer_init_zero (:obj:`bool`): Whether to use zero initializationss for the last layer of \
                dynamics/prediction mlp, default sets it to True.
            - state_norm (:obj:`bool`): Whether to use normalization for hidden states, default set it to False.
            - downsample (:obj:`bool`): Whether to do downsampling for observations in ``representation_network``, \
                defaults to True. This option is often used in video games like Atari. In board games like go, \
                we don't need this module.
            - activation (:obj:`Optional[nn.Module]`): Activation function used in network, which often use in-place \
                operation to speedup, e.g. ReLU(inplace=True).
            - norm_type (:obj:`str`): The type of normalization in networks. defaults to 'BN'.
            - discrete_action_encoding_type (:obj:`str`): The type of encoding for discrete action. Default sets it to 'one_hot'. 
                options = {'one_hot', 'not_one_hot'}
        """
        super(ObjectZeroRNNFullObsModel, self).__init__()
        self.categorical_distribution = categorical_distribution
        if self.categorical_distribution:
            self.reward_support_size = len(torch.arange(*reward_support_range))
            self.value_support_size = len(torch.arange(*value_support_range))
        else:
            self.reward_support_size = 1
            self.value_support_size = 1

        self.action_space_size = action_space_size
        assert discrete_action_encoding_type in ['one_hot', 'not_one_hot'], discrete_action_encoding_type
        self.discrete_action_encoding_type = discrete_action_encoding_type
        if self.discrete_action_encoding_type == 'one_hot':
            self.action_encoding_dim = action_space_size
        elif self.discrete_action_encoding_type == 'not_one_hot':
            self.action_encoding_dim = 1
        self.rnn_hidden_size = rnn_hidden_size
        self.proj_hid = proj_hid
        self.proj_out = proj_out
        self.pred_hid = pred_hid
        self.pred_out = pred_out
        self.last_linear_layer_init_zero = last_linear_layer_init_zero
        self.state_norm = state_norm
        self.downsample = downsample
        self.self_supervised_learning_loss = self_supervised_learning_loss
        self.norm_type = norm_type
        self.activation = activation
        self.analysis_sim_norm = analysis_sim_norm
        self.env_num = collector_env_num
        self.timestep = 0
        self.context_length_init = context_length_init
        self.last_ready_env_id = None

        latent_size = 512

        self.dynamics_network = OCDynamicsNetwork(
            observation_shape,
            self.action_encoding_dim,
            num_res_blocks,
            num_channels + self.action_encoding_dim,
            reward_head_channels,
            reward_head_hidden_channels,
            self.reward_support_size,
            flatten_input_size_for_reward_head,
            downsample,
            rnn_hidden_size,
            last_linear_layer_init_zero=self.last_linear_layer_init_zero,
            activation=activation,
            norm_type=norm_type,
            embedding_dim=768,
            group_size=8,
            use_sim_norm=use_sim_norm,  # NOTE
            res_connection_in_dynamics=True,
        )

        self.prediction_network = OCPredictionNetwork(
            observation_shape,
            action_space_size,
            num_res_blocks,
            num_channels,
            value_head_channels,
            policy_head_channels,
            value_head_hidden_channels,
            policy_head_hidden_channels,
            self.value_support_size,
            flatten_input_size_for_value_head,
            flatten_input_size_for_policy_head,
            downsample,
            last_linear_layer_init_zero=self.last_linear_layer_init_zero,
            activation=self.activation,
            norm_type=self.norm_type,
            gru_hidden_size=self.rnn_hidden_size

        )

        if self.self_supervised_learning_loss:
            if self.downsample:
                # In Atari, if the observation_shape is set to (12, 96, 96), which indicates the original shape of
                # (3,96,96), and frame_stack_num is 4. Due to downsample, the encoding of observation (latent_state) is
                # (64, 96/16, 96/16), where 64 is the number of channels, 96/16 is the size of the latent state. Thus,
                # self.projection_input_dim = 64 * 96/16 * 96/16 = 64*6*6 = 2304
                self.projection_input_dim = num_channels * latent_size
            else:
                self.projection_input_dim = num_channels * observation_shape[1] * observation_shape[2]

            self.projection = nn.Sequential(
                nn.Linear(self.projection_input_dim, self.proj_hid), nn.BatchNorm1d(self.proj_hid), activation,
                nn.Linear(self.proj_hid, self.proj_hid), nn.BatchNorm1d(self.proj_hid), activation,
                nn.Linear(self.proj_hid, self.proj_out), nn.BatchNorm1d(self.proj_out)
            )
            self.prediction_head = nn.Sequential(
                nn.Linear(self.proj_out, self.pred_hid),
                nn.BatchNorm1d(self.pred_hid),
                activation,
                nn.Linear(self.pred_hid, self.pred_out),
            )

    def initial_inference(self, last_obs: torch.Tensor, last_action=None, current_obs=None, ready_env_id=None,
                          last_ready_env_id=None) -> 'MZRNNNetworkOutput':
        """
        Perform initial inference based on the phase (training or evaluation/collect).

        Arguments:
            - last_obs (:obj:`torch.Tensor`): The last observation tensor.
            - last_action: The last action taken.
            - current_obs: The current observation tensor.
            - ready_env_id: The ready environment ID.
            - last_ready_env_id: The last ready environment ID.

        Returns:
            MZRNNNetworkOutput: The output object containing value, policy logits, and latent states.
        """
        if self.training or last_action is None:
            # ===================== Training phase  ======================
            batch_size = last_obs.shape[0]
            self.timestep = 0
            self.current_latent_state = self._representation(last_obs)

            # Initialize hidden state
            self.world_model_latent_history_init_complete = torch.zeros(1, batch_size, self.rnn_hidden_size).to(
                last_obs.device)

            # Compute prediction
            policy_logits, value = self._prediction(self.current_latent_state,
                                                    self.world_model_latent_history_init_complete)
            # NOTE: need to pass the gradient
            selected_world_model_latent_history = self.world_model_latent_history_init_complete
        else:
            #  ===================== Inference phase at Evaluation/Collect  =====================
            batch_size = current_obs.shape[0]

            if last_action is not None and max(last_action) == -1:
                # First step of an episode
                self.current_latent_state = self._representation(current_obs)
                self.world_model_latent_history_init_complete = torch.zeros(1, self.env_num, self.rnn_hidden_size).to(
                    last_obs.device)
                self.last_latent_state = self.current_latent_state
            else:
                # The second to last steps of an episode
                last_action = torch.from_numpy(np.array(last_action)).to(self.current_latent_state.device)
                self.last_latent_state = self._representation(last_obs)  # NOTE: note it's last_obs

                if len(last_ready_env_id) == self.env_num:
                    _, self.world_model_latent_history_init_complete, _ = self._dynamics(self.last_latent_state,
                                                                                         self.world_model_latent_history_init_complete,
                                                                                         last_action)
                else:
                    last_index_tensor = torch.tensor(list(last_ready_env_id))
                    self.world_model_latent_history_init = copy.deepcopy(
                        self.world_model_latent_history_init_complete[:, last_index_tensor, :])
                    _, self.world_model_latent_history_init, _ = self._dynamics(self.last_latent_state,
                                                                                self.world_model_latent_history_init,
                                                                                last_action)
                    self.world_model_latent_history_init_complete[:, last_index_tensor, :] = self.world_model_latent_history_init

                self.current_latent_state = self._representation(current_obs)

                if self.timestep % self.context_length_init == 0:
                    # TODO: the method that use the recent context, rather than the hard reset
                    self.world_model_latent_history_init_complete = torch.zeros(1, self.env_num, self.rnn_hidden_size).to(last_obs.device)

            if len(ready_env_id) == self.env_num:
                selected_world_model_latent_history = copy.deepcopy(self.world_model_latent_history_init_complete)
                policy_logits, value = self._prediction(self.current_latent_state, selected_world_model_latent_history)
            else:
                # the ready_env_id is not complete, need to select the corresponding latent history
                index_tensor = torch.tensor(list(ready_env_id))
                selected_world_model_latent_history = copy.deepcopy(self.world_model_latent_history_init_complete[:, index_tensor, :])
                policy_logits, value = self._prediction(self.current_latent_state, selected_world_model_latent_history)

        self.timestep += 1
        return MZRNNNetworkOutput(value, [0. for _ in range(batch_size)], policy_logits, self.current_latent_state, None,
                               selected_world_model_latent_history)

    def recurrent_inference(
            self,
            latent_state: torch.Tensor,
            world_model_latent_history: Tuple[torch.Tensor],
            action: torch.Tensor,
            next_latent_state: Optional[Tuple[torch.Tensor]] = None,
            ready_env_id: Optional[int] = None
    ) -> MZRNNNetworkOutput:
        """
        Perform recurrent inference to predict the next latent state, reward, and policy logits.

        Arguments:
            - latent_state (:obj:`torch.Tensor`): The current latent state tensor.
            - world_model_latent_history (:obj:`Tuple[torch.Tensor]`): The history of latent states from the world model.
            - action (:obj:`torch.Tensor`): The action tensor.
            - next_latent_state (:obj:`Optional[Tuple[torch.Tensor]], optional`): The next latent state tensor if available. Defaults to None.
            - ready_env_id (:obj:`Optional[int], optional`): ID of the ready environment. Defaults to None.

        Returns:
            MZRNNNetworkOutput: An object containing value, reward, policy logits, next latent state,
                             predicted next latent state, and updated world model latent history.
        """

        # Use the dynamics model to predict the next latent state and reward
        predict_next_latent_state, world_model_latent_history, reward = self._dynamics(
            latent_state, world_model_latent_history, action
        )

        # Determine which latent state to use for prediction
        inference_latent_state = next_latent_state if next_latent_state is not None else predict_next_latent_state

        # Use the prediction model to get policy logits and value
        policy_logits, value = self._prediction(inference_latent_state, world_model_latent_history)

        # If next_latent_state is provided, use it; otherwise, use the predicted next latent state
        return MZRNNNetworkOutput(value, reward, policy_logits, next_latent_state, predict_next_latent_state, world_model_latent_history)

    def _prediction(self, latent_state: torch.Tensor, world_model_latent_history: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """
        Overview:
             use the prediction network to predict the "value" and "policy_logits" of the "latent_state".
        Arguments:
            - latent_state (:obj:`torch.Tensor`): The encoding latent state of input state.
        Returns:
            - policy_logits (:obj:`torch.Tensor`): The output logit to select discrete action.
            - value (:obj:`torch.Tensor`): The output value of input state to help policy improvement and evaluation.
        Shapes:
            - latent_state (:obj:`torch.Tensor`): :math:`(B, H_, W_)`, where B is batch_size, H_ is the height of \
                latent state, W_ is the width of latent state.
            - policy_logits (:obj:`torch.Tensor`): :math:`(B, action_dim)`, where B is batch_size.
            - value (:obj:`torch.Tensor`): :math:`(B, value_support_size)`, where B is batch_size.
        """
        return self.prediction_network(latent_state, world_model_latent_history)

    def _dynamics(self, latent_state: torch.Tensor, world_model_latent_history: Tuple[torch.Tensor],
                  action: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor], torch.Tensor]:
        """
        Overview:
            Concatenate ``latent_state`` and ``action`` and use the dynamics network to predict ``next_latent_state``
            ``reward`` and ``next_world_model_latent_history``.
        Arguments:
            - latent_state (:obj:`torch.Tensor`): The encoding latent state of input state.
            - world_model_latent_history (:obj:`Tuple[torch.Tensor]`): The input hidden state of LSTM about reward.
            - action (:obj:`torch.Tensor`): The predicted action to rollout.
        Returns:
            - next_latent_state (:obj:`torch.Tensor`): The predicted latent state of the next timestep.
            - next_world_model_latent_history (:obj:`Tuple[torch.Tensor]`): The output hidden state of LSTM about reward.
            - reward (:obj:`torch.Tensor`): The predicted prefix sum of value for input state.
        Shapes:
            - latent_state (:obj:`torch.Tensor`): :math:`(B, H_, W_)`, where B is batch_size, H_ is the height of \
                latent state, W_ is the width of latent state.
            - action (:obj:`torch.Tensor`): :math:`(B, )`, where B is batch_size.
            - next_latent_state (:obj:`torch.Tensor`): :math:`(B, H_, W_)`, where B is batch_size, H_ is the height of \
                latent state, W_ is the width of latent state.
            - reward (:obj:`torch.Tensor`): :math:`(B, reward_support_size)`, where B is batch_size.
        """
        # NOTE: the discrete action encoding type is important for some environments

        # discrete action space
        if self.discrete_action_encoding_type == 'one_hot':
            # Stack latent_state with the one hot encoded action.
            # The final action_encoding shape is (batch_size, action_space_size, latent_state[2], latent_state[3]), e.g. (8, 2, 4, 1).
            if len(action.shape) == 1:
                # (batch_size, ) -> (batch_size, 1)
                # e.g.,  torch.Size([8]) ->  torch.Size([8, 1])
                action = action.unsqueeze(-1)

            # transform action to one-hot encoding.
            # action_one_hot shape: (batch_size, action_space_size), e.g., (8, 4)
            action_one_hot = torch.zeros(action.shape[0], self.action_space_size, device=action.device)
            # transform action to torch.int64
            action = action.long()
            action_one_hot.scatter_(1, action, 1)

            action_encoding_tmp = action_one_hot.unsqueeze(-1).unsqueeze(-1)
            action_encoding = action_encoding_tmp.expand(
                latent_state.shape[0], self.action_space_size, latent_state.shape[2], latent_state.shape[3]
            )

        elif self.discrete_action_encoding_type == 'not_one_hot':
            # Stack latent_state with the normalized encoded action.
            # The final action_encoding shape is (batch_size, 1, latent_state[2], latent_state[3]), e.g. (8, 1, 4, 1).
            if len(action.shape) == 2:
                # (batch_size, action_dim=1) -> (batch_size, 1, 1, 1)
                # e.g.,  torch.Size([8, 1]) ->  torch.Size([8, 1, 1, 1])
                action = action.unsqueeze(-1).unsqueeze(-1)
            elif len(action.shape) == 1:
                # (batch_size,) -> (batch_size, 1, 1, 1)
                # e.g.,  -> torch.Size([8, 1, 1, 1])
                action = action.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

            action_encoding = action.expand(
                latent_state.shape[0], 1, latent_state.shape[2], latent_state.shape[3]
            ) / self.action_space_size

        # state_action_encoding shape: (batch_size, latent_state[1] + action_dim, latent_state[2], latent_state[3]) or
        # (batch_size, latent_state[1] + action_space_size, latent_state[2], latent_state[3]) depending on the discrete_action_encoding_type.
        state_action_encoding = torch.cat((latent_state, action_encoding), dim=1)

        # NOTE: the key difference between MuZeroRNN and MuZero
        next_latent_state, next_world_model_latent_history, reward = self.dynamics_network(
            state_action_encoding, world_model_latent_history
        )

        if self.state_norm:
            next_latent_state = renormalize(next_latent_state)
        return next_latent_state, next_world_model_latent_history, reward