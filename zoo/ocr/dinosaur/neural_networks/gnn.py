import torch

import torch.nn as nn

from zoo.ocr.dinosaur.neural_networks import utils


class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, act_fn='relu', layer_norm=True, num_layers=3, use_interactions=True,
                 output_dim=None):
        super(GNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        if self.output_dim is None:
            self.output_dim = self.input_dim

        self.use_interactions = use_interactions
        self.num_layers = num_layers

        edge_mlp_input_size = self.input_dim * 2

        self.edge_mlp = nn.Sequential(*self.make_node_mlp_layers_(
            edge_mlp_input_size, self.hidden_dim, act_fn, layer_norm
        ))

        if not self.use_interactions:
            node_input_dim = self.input_dim
        else:
            node_input_dim = hidden_dim + self.input_dim

        self.node_mlp = nn.Sequential(*self.make_node_mlp_layers_(
            node_input_dim, self.output_dim, act_fn, layer_norm
        ))

        self.edge_list = None
        self.batch_size = 0

    def _edge_model(self, source, target):
        x = [source, target]
        out = torch.cat(x, dim=1)
        return self.edge_mlp(out)

    def _node_model(self, node_attr, edge_index, edge_attr):
        if edge_attr is not None:
            row, col = edge_index
            agg = utils.unsorted_segment_sum(
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

    def forward(self, states):
        device = states.device
        batch_size = states.size(0)
        num_nodes = states.size(1)

        # states: [batch_size (B), num_objects, embedding_dim]
        # node_attr: Flatten states tensor to [B * num_objects, embedding_dim]
        node_attr = states.reshape(-1, self.input_dim)
        edge_attr = None
        edge_index = None

        if num_nodes > 1 and self.use_interactions:
            # edge_index: [B * (num_objects*[num_objects-1]), 2] edge list
            edge_index = self._get_edge_list_fully_connected(
                batch_size, num_nodes, device)

            row, col = edge_index
            edge_attr = self._edge_model(node_attr[row], node_attr[col])

        node_attr = self._node_model(
            node_attr, edge_index, edge_attr)

        # [batch_size, num_nodes, hidden_dim]
        node_attr = node_attr.view(batch_size, num_nodes, -1)

        # we return the same thing as input but with a changed state
        # this allows us to stack GNNs
        return node_attr

    def make_node_mlp_layers_(self, input_dim, output_dim, act_fn, layer_norm):
        return utils.make_node_mlp_layers(self.num_layers, input_dim, self.hidden_dim, output_dim, act_fn, layer_norm)


class GRUGNNCell(nn.Module):
    def __init__(self, update_dim, slot_dim, gnn_hidden_dim, act=torch.tanh, update_bias=-1):
        super(GRUGNNCell, self).__init__()
        self._update_dim = update_dim
        self._slot_dim = slot_dim
        self._gnn_hidden_dim = gnn_hidden_dim
        self._act = act
        self._update_bias = update_bias
        self._gnn = GNN(self._update_dim + self._slot_dim, self._gnn_hidden_dim, layer_norm=True, num_layers=3,
                        use_interactions=True, output_dim=3 * self._slot_dim)

    def forward(self, updates, slots_prev):
        node = torch.cat([updates, slots_prev], dim=-1)
        x = self._gnn(node)
        reset, cand, update = torch.split(x, [self._slot_dim] * 3, dim=-1)
        reset = torch.sigmoid(reset)
        cand = self._act(reset * cand)
        update = torch.sigmoid(update + self._update_bias)
        slots = update * cand + (1 - update) * slots_prev
        return slots
