import sys
sys.path.insert(0, '../')

import gym
import gym_sokoban
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils.convert import to_networkx
from torch_geometric.utils import subgraph
from torch_geometric.utils import to_dense_adj
import data.dataset
from data.constants import TinyWorldElements as elem
from data.embedding import MinimalEmbedding, NoWallsEmbedding, NoWallsV2Embedding
from data.graph_env import GraphEnv
from torch_geometric.data import Data

dset = data.dataset.InMemorySokobanDataset(
    root='levels/very_easy_1/train',
    embedding=MinimalEmbedding())#'tiny_rgb_array')

state = dset.init_states[1]
img = dset.init_images[1]
env=GraphEnv(embedding=MinimalEmbedding(), device='cpu')
nonwall_indices=torch.argwhere(state.x[:,-1]!=1).flatten()
nonwall_mask=state.x[:,-1]!=1
x_reduced = state.x[nonwall_mask]
ei_reduced, _= subgraph(nonwall_indices,state.edge_index, relabel_nodes=True)
A = to_dense_adj(ei_reduced).squeeze()
pos_reduced=state.pos[nonwall_indices]
new_player_idx = torch.argwhere(x_reduced[:,1]==1).flatten()
new_neighbor_mask=A[new_player_idx].to(torch.bool)
reduced_state = Data(
    x=x_reduced,
    pos=pos_reduced,
    edge_index=ei_reduced,
    mask=new_neighbor_mask,
    player_idx=new_player_idx, A=A
    )

env.define(init_state=reduced_state)
env.render()
plt.imshow(img)
plt.savefig('test.png')

#for move in [7,6,5,6,7,8,13,12,11,10,11]:
for move in [7,6,11,10,5,6,2,1,5,10,11,12,11,10]:
    s,r,d,i=env.step(torch.tensor([move]))
    print('moved_to',move,'reward=',r,'done=',d)
    env.render()

k=0



# #env = gym.make("Sokoban-small-v1")
# from gym_sokoban.envs.sokoban_env import SokobanEnv
# env=SokobanEnv(dim_room=(5,5),max_steps=50,num_boxes=1)
# s=env.reset(render_mode="tiny_rgb_array")
# emb = MinimalEmbedding()
# # 2 different displays
# def show_grids():
#     fix, axes = plt.subplots(1, 2)
#     axes[0].imshow(env.render("rgb_array"))
#     axes[1].imshow(env.render("tiny_rgb_array"))
#     plt.savefig("test.png")

# def show_graph_nx(graph_nx, node_color, pos_map, title=""):
#     plt.figure(1, figsize=(7,7)) 
#     plt.title(title)
#     nx.draw(graph_nx, cmap=plt.get_cmap('seismic'), node_color=node_color, node_size=75, linewidths=6, pos=pos_map)
#     plt.show()


while True:    
    show_grids()
    l = env.get_action_lookup()
    print(l)
    m = int(input("move:"))
    s,r,d,i=env.step(m, observation_mode="tiny_rgb_array")
    G=emb(s)
    pos_map = {i: pos.numpy() for i, pos in enumerate(G.pos)}
    show_graph_nx(to_networkx(G), G.x[:, 0], pos_map, "Boxes")
    show_graph_nx(to_networkx(G), G.x[:, 1], pos_map, "Player")
    show_graph_nx(to_networkx(G), G.x[:, 2], pos_map, "Target")
    show_graph_nx(to_networkx(G), G.x[:, 3], pos_map, "Walls")
    show_graph_nx(to_networkx(G), G.x[:, 4], pos_map, "Void")
    print('reward',r)