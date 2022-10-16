import copy
from pathlib import Path
from typing_extensions import Self
import numpy as np
import time
from torch_geometric.nn.conv import MessagePassing, GATv2Conv
from torch_geometric.nn.models.basic_gnn import BasicGNN
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_adj
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class GATv2(BasicGNN):
    r"""
    """
    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:

        kwargs = copy.copy(kwargs)
        if 'heads' in kwargs and out_channels % kwargs['heads'] != 0:
            kwargs['heads'] = 1
        if 'concat' not in kwargs or kwargs['concat']:
            out_channels = out_channels // kwargs.get('heads', 1)
        return GATv2Conv(in_channels, out_channels, dropout=self.dropout,
                       **kwargs)


class PPO_GNN_Model(nn.Module):
    # Base class for PPO-GNN-LSTM implementations
    def __init__(self, tp):
        self.tp=tp
        super(PPO_GNN_Model, self).__init__()
        self.data = []
        self.ei = []
        #self.obs_space = env.observation_space.shape[0]
        #self.act_space = env.action_space
        #self.num_nodes = env.action_space.n
        self.node_dim = 3
        self.emb_dim = 24
        self.num_rollouts = 2
        self.num_epochs = 2
        Path(self.tp["base_checkpoint_path"]).mkdir(parents=True, exist_ok=True)

    def _deserialize(self, obs):
        # Deconstruct from an observation as defined by the PPO flattened observation wrapper
        # Args: obs [ nfm (NxF) | edge_list (Ex2) | reachable (N,) | num_nodes (1,) | max_num_nodes (1,) | num_edges (1,) | max_num_edges (1,) | node_dim (1,)]
        num_nodes, max_nodes, num_edges, max_edges, node_dim = obs[-5:].to(torch.int64).tolist()
        nfm , edge_index , reachable, _ = torch.split(obs,(node_dim * max_nodes, 2 * max_edges, max_nodes, 5), dim=0)
        nfm = nfm.reshape(max_nodes,-1)[:num_nodes]
        edge_index = edge_index.reshape(2,-1)[:,:num_edges].to(torch.int64)
        reachable = reachable.reshape(-1,1).to(torch.int64)[:num_nodes]
        return nfm, edge_index, reachable, num_nodes, max_nodes, num_edges, max_edges, node_dim

    def put_data(self, transition_buffer, ei=None):
        self.data.append(transition_buffer)
        self.ei.append(ei)

    def checkpoint(self, n_epi, mean_Return, mode=None):
        if mode == 'best':
            print('...New best det results, saving model, it=',n_epi,'avg_return=',mean_Return)
            fname = self.tp["base_checkpoint_path"] + "best_model.tar"
            announce = 'BEST_' if mode=='best' else '...._'
            OF = open(self.tp["base_checkpoint_path"]+'/model_best_save_history.txt', 'a')
            OF.write(announce+'timestep:'+str(n_epi)+', avg det res:'+str(mean_Return)+'\n')
            OF.close()            
        elif mode == 'last':
            fname = self.tp["base_checkpoint_path"] + "model_"+str(n_epi)+".tar"
        else: return
        #checkpoint = torch.load(fname)
        #self.load_state_dict(checkpoint['weights'])
        torch.save({
            'weights':self.state_dict(),
            'optimizer':self.optimizer.state_dict(),
        }, fname)

    def numTrainableParameters(self):
        ps=""
        ps+=self.description+'\n'
        ps+='------------------------------------------\n'
        total = 0
        for name, p in self.named_parameters():
            if p.requires_grad:
                total += np.prod(p.shape)
            ps+=("{:24s} {:12s} requires_grad={}\n".format(name, str(list(p.shape)), p.requires_grad))
        ps+=("Total number of trainable parameters: {}\n".format(total))
        ps+='------------------------------------------'
        assert total == sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, ps


class PPO_GNN_NoLSTM(PPO_GNN_Model):
    # Single LSTM acting either on node features (lstm_type='FE') or on embedding ('EMB'). LSTM can be disabled ('None')
    def __init__(self, tp):
        super(PPO_GNN_NoLSTM, self).__init__(tp)
        self.description="PPO policy, GATv2 extractor, no lstm, action masking"
        self.DISCOUNT=0.98
        self.GAE_LAMBDA=0.95
        self.PPO_CLIP=0.2
        self.CHECKPOINT_FREQ=100
        print('device',device)
        kwargs={'concat':True}
        self.gat = GATv2(
            in_channels = self.node_dim,
            hidden_channels = self.emb_dim,
            heads = 3,
            num_layers = 5,
            out_channels = self.emb_dim,
            share_weights = False,
            **kwargs
        ).to(device) 
        self.gat.supports_edge_weight=False
        self.gat.supports_edge_attr=False    
      
        # Policy network parameters
        self.theta5_pi = nn.Linear(2*self.emb_dim, 1, True, device=device)#, dtype=torch.float32)
        self.theta6_pi = nn.Linear(self.emb_dim, self.emb_dim, True, device=device)#, dtype=torch.float32)
        self.theta7_pi = nn.Linear(self.emb_dim, self.emb_dim, True, device=device)#, dtype=torch.float32)

        self.theta5_v = nn.Linear(2*self.emb_dim, 1, True, device=device)#, dtype=torch.float32)
        self.theta6_v = nn.Linear(self.emb_dim, self.emb_dim, True, device=device)#, dtype=torch.float32)
        self.theta7_v = nn.Linear(self.emb_dim, self.emb_dim, True, device=device)#, dtype=torch.float32)

        self.optimizer = optim.Adam(self.parameters(), lr=2e-4)

    def process_input(self, nfm, ei, reachable):
        nfm = nfm.to(device)
        reachable = reachable.to(device)
        ei = ei.to(device)
        if len(nfm.shape) == 2:
            nfm = nfm[None,:,:]
            reachable = reachable[None,:]
        seq_len = nfm.shape[0]
        num_nodes = nfm.shape[1]

        return num_nodes, seq_len, nfm, ei, reachable

    def create_embeddings(self, seq_len, nfm, ei):
        pyg_list=[]
        for e in nfm:
            pyg_list.append(Data(e, ei))
        pyg_data = Batch.from_data_list(pyg_list)
        mu = self.gat(pyg_data.x, pyg_data.edge_index)
        
        mu = mu.reshape(seq_len, -1, self.emb_dim) # mu: (seq_len, num_nodes, emb_dim)        
        
        return mu

    def pi(self, nfm, ei, reachable):
        num_nodes, seq_len, nfm, ei, reachable = self.process_input(nfm, ei, reachable)
        mu = self.create_embeddings(seq_len, nfm, ei)
        
        mu_meanpool = mu.mean(dim=1, keepdim=True) # mu_meanpool: (seq_len, 1, emb_dim). scatter mean not needed: equal size graphs
        
        expander = torch.tensor([num_nodes]*seq_len, dtype=torch.int64, device=device)
        mu_meanpool_expanded = torch.repeat_interleave(mu_meanpool, expander, dim=0).reshape(seq_len, num_nodes, self.emb_dim) # (seq_len, num_nodes, emb_dim) # (seq_len, num_nodes, emb_dim)
        global_state = self.theta6_pi(mu_meanpool_expanded) # yields (seq_len, nodes in batch, emb_dim)
        local_action = self.theta7_pi(mu)  # yields (seq_len, nodes in batch, emb_dim)
        rep = F.relu(torch.cat((global_state, local_action), dim=2)) # concat creates (#nodes in batch, 2*emb_dim)        
        
        prob_logits = self.theta5_pi(rep).squeeze(-1) # (nr_nodes in batch,)
        prob_logits[~reachable] = -torch.inf
        prob = F.softmax(prob_logits, dim=1)

        return prob
    
    def v(self, nfm, ei, reachable):
        num_nodes, seq_len, nfm, ei, reachable = self.process_input(nfm, ei, reachable)
        mu = self.create_embeddings(seq_len, nfm, ei)
        
        mu_meanpool = mu.mean(dim=1, keepdim=True) # mu_meanpool: (seq_len, 1, emb_dim). scatter mean not needed: equal size graphs

        expander = torch.tensor([num_nodes]*seq_len, dtype=torch.int64, device=device)
        mu_meanpool_expanded = torch.repeat_interleave(mu_meanpool, expander, dim=0).reshape(seq_len,num_nodes,self.emb_dim) # (seq_len, num_nodes, emb_dim)
        global_state = self.theta6_v(mu_meanpool_expanded) # yields (seq_len, nodes in batch, emb_dim)
        local_action = self.theta7_v(mu)  # yields (seq_len, nodes in batch, emb_dim)
        rep = F.relu(torch.cat([global_state, local_action], dim=2)) # concat creates (#nodes in batch, 2*emb_dim)
        
        qvals = self.theta5_v(rep).squeeze(-1)  #(seq_len, nr_nodes in batch)
        qvals[~reachable] = -torch.inf
        v = qvals.max(dim=1)[0].unsqueeze(-1)
        
        return v
         
    def make_batch(self):
        batch = []
        for i in range(self.num_rollouts):
            s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst, reachable_lst, reachable_prime_lst = [], [], [], [], [], [], [], []
            for transition in self.data[i]:
                s, a, r, s_prime, prob_a, done, reachable, reachable_prime = transition
                
                s_lst.append(s)
                a_lst.append([a])
                r_lst.append([r])
                s_prime_lst.append(s_prime)
                prob_a_lst.append([prob_a])
                done_mask = 0 if done else 1
                done_lst.append([done_mask])
                reachable_lst.append(reachable)
                reachable_prime_lst.append(reachable_prime)
                
            s,a,r,s_prime,done_mask,prob_a,reachable,reachable_prime = torch.stack(s_lst), torch.tensor(a_lst), \
                                            torch.tensor(r_lst), torch.stack(s_prime_lst), \
                                            torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst), \
                                            torch.stack(reachable_lst), torch.stack(reachable_prime_lst)
            batch.append((s, a, r, s_prime, done_mask, prob_a, reachable, reachable_prime, self.ei[i]))
        self.data = []
        self.ei=[]
        return batch#s, a, r, s_prime, done_mask, prob_a,  h_in_lst[0], h_out_lst[0], mask
        
    def train_net(self, n_epi=0):
        batch = self.make_batch()
        rlist, l1list, l2list ,l3list, ltlist = [],[],[],[],[]
        for _ in range(self.num_epochs):
            mean_ratio_tsr = torch.tensor([])
            mean_loss1_tsr = torch.tensor([])
            mean_loss2_tsr = torch.tensor([])
            mean_loss3_tsr = torch.tensor([])
            loss_tsr = torch.tensor([])
            for j in range(self.num_rollouts):
                nfm, a, r, nfm_prime, done_mask, prob_a, reachable, reachable_prime, ei = batch[j]
                

                v_prime = self.v(nfm_prime, ei, reachable_prime).cpu()
                td_target = r + self.DISCOUNT * v_prime * done_mask
                v_s = self.v(nfm, ei, reachable).cpu()
                delta = td_target - v_s
                delta = delta.detach().numpy()

                advantage_lst = []
                advantage = 0.0
                for item in delta[::-1]:
                    advantage = self.DISCOUNT * self.GAE_LAMBDA * advantage + item[0]
                    advantage_lst.append([advantage])
                advantage_lst.reverse()
                advantage = torch.tensor(advantage_lst, dtype=torch.float)

                pi = self.pi(nfm, ei, reachable)
                pi=pi.cpu()
                pi_a = pi.squeeze(1).gather(1,a)
                ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.PPO_CLIP, 1 + self.PPO_CLIP) * advantage
                
                loss1 = -torch.min(surr1, surr2)
                loss2 = F.smooth_l1_loss(v_s , td_target.detach())
                loss3 = -torch.distributions.Categorical(probs=pi).entropy()
                loss = loss1 + loss2 # note: We fix c1=1, c2=0 (see PPO paper)
                #loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(v_s , td_target.detach())
                loss_tsr = torch.concat((loss_tsr,loss.squeeze(-1)))
                
                # log individual loss components
                mean_ratio_tsr = torch.concat((mean_ratio_tsr,ratio.detach().cpu().mean().unsqueeze(-1))) 
                mean_loss1_tsr = torch.concat((mean_loss1_tsr,loss1.detach().cpu().mean().unsqueeze(-1))) 
                mean_loss2_tsr = torch.concat((mean_loss2_tsr,loss2.detach().cpu().unsqueeze(-1))) 
                mean_loss3_tsr = torch.concat((mean_loss3_tsr,loss3.detach().cpu().mean().unsqueeze(-1))) 
            ratio_=mean_ratio_tsr.mean()
            loss1_=mean_loss1_tsr.mean()
            loss2_=mean_loss2_tsr.mean()
            loss3_=mean_loss3_tsr.mean()
            
            self.optimizer.zero_grad()
            loss_tsr.mean().backward(retain_graph=True)
            #torch.nn.utils.clip_grad.clip_grad_norm_(self.parameters(), 0.5)
            self.optimizer.step()
            
            rlist.append(ratio_.item())
            l1list.append(loss1_.item())
            l2list.append(loss2_.item())
            l3list.append(loss3_.item())
            ltlist.append(loss_tsr.mean().detach().cpu().item())
        self.tp['writer'].add_scalar('ratio', np.mean(rlist), n_epi)
        self.tp['writer'].add_scalar('loss1_ratio', np.mean(l1list), n_epi)
        self.tp['writer'].add_scalar('loss2_value', np.mean(l2list), n_epi)
        self.tp['writer'].add_scalar('loss3_entropy', np.mean(l3list), n_epi)
        self.tp['writer'].add_scalar('loss_total', np.mean(ltlist), n_epi)

    def learn(self, senv, it0=0, best_result=-1e6):
        score = 0.0
        counter = 0
        validation_interval = 50
        current_max_Return  = best_result

        for n_epi in range(it0, 10000):
            env=senv[0]
            R=0
            gathertimes=[]
            traintimes=[]
            start_gather_time = time.time()
            for t in range(self.num_rollouts):
                s = env.reset()
                
                done = False
                transitions=[]
                
                while not done:
                    #mask = env.action_masks()
                    reachable=s.A[s.player_idx].squeeze().to(torch.bool)
                    prob = self.pi(s.x, s.edge_index, reachable)
                    prob = prob.view(-1)
                    m = Categorical(prob)
                    a = m.sample().item()
                    s_prime, r, done, info = env.step(a)
                    #print('move',a, 'reward',r)
                    #env.render()
                    
                    reachable_prime=s_prime.A[s.player_idx].squeeze().to(torch.bool)
                    transitions.append((s.x, a, r/10.0, s_prime.x, prob[a].item(), done, reachable, reachable_prime))
                    s = s_prime

                    score += r
                    R += r
                    if done:
                        break
                self.put_data(transitions, s.edge_index)
                transitions = []
            counter+=1
            end_gather_time = time.time()
            start_train_time = time.time()
            self.train_net(n_epi)
            end_train_time = time.time() 
            gathertimes.append(end_gather_time-start_gather_time)
            traintimes.append(end_train_time-start_train_time)

            self.tp['writer'].add_scalar('return_per_epi', R/self.num_rollouts, n_epi)
            
            if n_epi % self.CHECKPOINT_FREQ==0 and n_epi != it0:
                self.checkpoint(n_epi,score/counter/self.num_rollouts,mode='last')
            if n_epi % validation_interval == 0 and n_epi != it0:
                mean_Return = score/validation_interval/self.num_rollouts
                if mean_Return >= current_max_Return:            
                    current_max_Return = mean_Return
                    self.checkpoint(n_epi,mean_Return,mode='best')
                print("# of episode :{}, avg score : {:.1f}, gather time per iter: {:.1f}, train time per iter: {:.1f}".format(n_epi, mean_Return, np.mean(gathertimes), np.mean(traintimes)))
                counter=0
                score = 0.0
        #env.close()
        return mean_Return