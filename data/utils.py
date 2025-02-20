import copy
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx

from data.constants import ASCII_TO_PIXELS, MEANING_TO_COLOR, MEANING_TO_STATE


def ascii_to_img(filepath):
    """Convert a level saved in .txt format to an RGB image.
    Args: 
        - filepath : str, path to the saved .txt file
    Returns:
        - state_pixels : (W, H, 3) np.uint8  
    """
    with open(filepath, "r") as f:
        pixels = []
        for line in f:
            pix = torch.cat([ASCII_TO_PIXELS[s][None] for s in line if s != "\n"])[None]
            pixels.append(pix)
        pixels = torch.cat(pixels, 0)

    return pixels.numpy()


def all_boxes_on_all_targets(state):
    return all(state.x[state.x[:, 2] == 1][:, 0])


def is_neighbor_of_player(node_idx, mask):
    """Check if a node is in the neighborhood of the player"""
    player_neighbors = mask.nonzero()
    return node_idx in player_neighbors


def find_player_idx(x):
    _player_idx = (x[:, 1] == 1).nonzero().item()
    player_idx = torch.tensor(_player_idx).long().unsqueeze(0)
    return player_idx


def clone_and_detach_data(state):
    """Clone and detach a torch_gemetric.data.Data"""
    return Data.from_dict(
        {
            k: v.clone().detach() if torch.is_tensor(v) else copy.deepcopy(v)
            for k, v in state.__dict__['_store'].items()
        }
    )


def are_off_target_boxes_in_corner(state):
    """Verify if any off target boxes is stuck in a corner

    This is useful to stop the game early and/or penalize

    BUG (!!!) : it does not differentiate a corner and a lane
    """
    # Gather boxes not on targets
    boxes_idx = ((state.x[:, 0] == 1) & (state.x[:, 2] == 0)).nonzero().view(-1)
    nb_boxes = boxes_idx.size(0)

    # Count number of walls in their neighbors
    nb_walls = 0
    for box_idx in boxes_idx:
        neighbors_idx = state.edge_index[:, state.edge_index[0] == box_idx][1]
        nb_walls += (state.x[neighbors_idx, 3] == 1).sum().item()

    # Theshold
    is_deadlock = nb_walls > nb_boxes

    return is_deadlock


def direction_to_node_idx(state, direction):
    """Direction to node index

    Args:
        - state: require at least DirectionalEmbedding
        - direction: long tensor (on same device as state) in {0, ..., 3}

    Returns:
        - node_idx: long tensor (on same device as state) of shape (1,) or (0,)
    """
    # Get neighbors edges indexes (directed from player_idx)
    nb_edge_idxs = (state.edge_index[0, :] == state.player_idx).nonzero()

    # Get idx corresponding to given direction (if possible)
    dir_edge_idx = nb_edge_idxs[state.edge_attr[nb_edge_idxs, direction] == 1]
    node_idx = state.edge_index[1, dir_edge_idx]
    return node_idx


def count_boxes(state):
    return (state.x[:, 0] == 1).sum().item()


def display_graph(state, q_values=None, node_size=3000):
    """display a graph state using networkx."""
    pos_map = {i: pos.numpy() for i, pos in enumerate(state.pos.cpu())}

    # Swap x, y, invert y
    pos_map = {i: np.array([x, y]) for i, (y, x) in pos_map.items()}
    max_y = max([y for x, y in pos_map.values()])
    pos_map = {i: np.array([x, max_y - y]) for i, (x, y) in pos_map.items()}

    # node color
    features = state.x.cpu()[:, :4]
    colors = np.empty((features.size(0), 3))
    types = np.empty(features.size(0), dtype=str)

    for i, m in enumerate(MEANING_TO_COLOR.keys()):
        temp = torch.all(features == MEANING_TO_STATE[m], -1).numpy()
        colors[temp] = MEANING_TO_COLOR[m]
        types[temp] = m

    # display q values on each node
    if q_values is not None:
        labels = {
            i: f"{types[i]} [{i}] \n{value.item():.5f}"
            for i, value in enumerate(q_values)
        }
    else:
        labels = {i: "{} [{}] ".format(types[i], i) for i in range(len(state.x))}

    nx.draw(
        to_networkx(state),
        node_color=colors,
        labels=labels,
        node_size=node_size,
        linewidths=1,
        font_color="w",
        font_size=10,
        pos=pos_map,
    )


def plot_history(history, history_eval=None, figsize=(10, 10)):
    plt.style.use("ggplot")

    fig, axes = plt.subplots(4, 1, figsize=figsize)
    # Mean cumulative reward
    axes[0].plot(history["mean_cum_reward"], lw=2, label="train")
    axes[0].set_ylabel("mean cum reward")

    # Mean loss
    axes[1].plot(history["mean_loss"], lw=2, label="train")
    axes[1].set_ylabel("mean loss")

    # Number of level solved (%)
    n_train = history["dataset size"][0]
    train_solved_norm = [100 * s / n_train for s in history["solved"]]
    axes[2].plot(train_solved_norm, lw=2, label="train")
    axes[2].set_ylabel("level solved (%)")

    # Epsilon
    axes[3].plot(history["epsilon"], lw=2, label="train")
    axes[3].set_ylabel("epsilon")

    if history_eval is not None:
        # Mean cumulative reward
        axes[0].plot(history_eval["mean_cum_reward"], lw=2, label="eval")
        axes[0].legend()

        # Number of level solved (%)
        n_eval = history_eval["dataset size"][0]
        eval_solved_norm = [100 * s / n_eval for s in history_eval["solved"]]
        axes[2].plot(eval_solved_norm, lw=2, label="eval")
        axes[2].legend()

    plt.xlabel("epochs")

    # plt.show()
