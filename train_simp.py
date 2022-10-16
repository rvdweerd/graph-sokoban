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
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
import random

from ppo import PPO_GNN_NoLSTM

def get_trainset():
    dset = data.dataset.InMemorySokobanDataset(
        root='levels/very_easy_1/train',
        embedding=MinimalEmbedding())#'tiny_rgb_array')

    state = dset.init_states[1]
    img = dset.init_images[1]
    env=GraphEnv(embedding=MinimalEmbedding(), device='cpu')
    nonwall_indices=torch.argwhere(state.x[:,-1]!=1).flatten()
    nonwall_mask=state.x[:,-1]!=1
    x_reduced = state.x[nonwall_mask][:,:3]
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
    senv=[env]
    #env.render()
    #plt.imshow(img)
    #plt.savefig('test.png')
    return senv

def main(args):
    #config, hp, tp = GetConfigs(args, suffix='simp')   
    tp={}
    hp={}
    config={}
    ##### TRAIN FUNCTION #####
    if True:#config['train']:
        senv = get_trainset()
        
        #if config['demoruns']:
        #    pass
        #    while True:
        #        break
        
        for seed in [0]:#config['seedrange']:
            #seed_everything(seed)
            #senv.seed(seed)
            logdir_='test'#config['logdir']+'/SEED'+str(seed)
            tp['writer'] = SummaryWriter(log_dir=f"{logdir_}/logs")
            tp["base_checkpoint_path"]=f"{logdir_}/checkpoints/"
            tp["seed_path"]=logdir_

            model = PPO_GNN_NoLSTM(tp)
            #last_checkpoint, it0, best_result = get_last_checkpoint_filename(tp)
            # if last_checkpoint is not None:
            #     cp = torch.load(last_checkpoint)
            #     model.load_state_dict(cp['weights'])
            #     model.optimizer.load_state_dict(cp['optimizer'])
            #     print(f"Loaded model from {last_checkpoint}")
            #     print('Iteration:', it0, 'best_result:', best_result)
            #try:
            score = model.learn(senv)
            #except:
            #    pass#continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)    
    #parser.add_argument('--train_on', default='None', type=str)
    parser.add_argument('--num_step', default=10000, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--recurrent_seq_len', default=2, type=int)
    parser.add_argument('--parallel_rollouts', default=4, type=int)
    parser.add_argument('--rollout_steps', default=100, type=int)
    #parser.add_argument('--patience', default=500, type=int)
    parser.add_argument('--checkpoint_frequency', default=5000, type=int)
    #parser.add_argument('--obs_mask', default='None', type=str, help='U observation masking type', choices=['None','freq','prob','prob_per_u','mix'])
    #parser.add_argument('--obs_rate', default=1.0, type=float)
    #parser.add_argument('--eval_rate', default=1.0, type=float)
    parser.add_argument('--emb_dim', default=24, type=int)
    parser.add_argument('--lstm_type', default='None', type=str, choices=['None','EMB','FE','Dual','DualCC'])
    #parser.add_argument('--lstm_hdim', default=24, type=int)
    #parser.add_argument('--lstm_layers', default=1, type=int)
    #parser.add_argument('--lstm_dropout', default=0.0, type=float)
    parser.add_argument('--emb_iterT', default=5, type=int)
    #parser.add_argument('--nfm_func', default='NFM_ev_ec_t_dt_at_um_us', type=str)
    parser.add_argument('--qnet', default='gat2', type=str)
    parser.add_argument('--critic', default='q', type=str, choices=['q','v']) # q=v value route, v=single value route
    parser.add_argument('--train', type=lambda s: s.lower() in ['true', 't', 'yes', '1'],default=False)
    parser.add_argument('--eval', type=lambda s: s.lower() in ['true', 't', 'yes', '1'],default=False)   
    parser.add_argument('--test', type=lambda s: s.lower() in ['true', 't', 'yes', '1'],default=False)       
    #parser.add_argument('--test_heur', type=lambda s: s.lower() in ['true', 't', 'yes', '1'],default=False)       
    parser.add_argument('--num_seeds', default=1, type=int)
    parser.add_argument('--seed0', default=0, type=int)
    parser.add_argument('--demoruns', type=lambda s: s.lower() in ['true', 't', 'yes', '1'],default=False)
    parser.add_argument('--eval_deter', type=lambda s: s.lower() in ['true', 't', 'yes', '1'],default=True)
    #parser.add_argument('--type_obs_wrap', default='BasicDict', type=str)
    args=parser.parse_args()
    main(args)