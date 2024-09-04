import torch
import numpy as np
import hydra
import randomname
import random
import matplotlib.pyplot as plt
import pickle
import gzip
import scipy.io as sco

from lib.utils.tokenizers import str_to_tokens, tokens_to_str
from omegaconf import OmegaConf, DictConfig
from collections.abc import MutableMapping

from torch.distributions import Categorical
from tqdm import tqdm

#from comet_ml import Experiment

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def init_run(cfg):
    trial_id = cfg.trial_id
    if cfg.job_name is None:
        cfg.job_name = '_'.join(randomname.get_name().lower().split('-') + [str(trial_id)])
    cfg.seed = random.randint(0, 100000) if cfg.seed is None else cfg.seed
    set_seed(cfg.seed)
    cfg = OmegaConf.to_container(cfg, resolve=True)  # Resolve config interpolations
    cfg = DictConfig(cfg)
    # logger.write_hydra_yaml(cfg)

    print(OmegaConf.to_yaml(cfg))
    with open('hydra_config.txt', 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))
    return cfg

def flatten_config(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def get_distribution_plot(data):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.set_xlabel("-Energy")
    ax.set_ylabel("Frequency")
    ax.hist(data, bins=100)
    return fig

class GFN:
    def __init__(self, cfg, tokenizer):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.setup_vars()
        self.init_policy()

    def setup_vars(self):
        cfg = self.cfg
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # Task stuff
        self.max_len = cfg.max_len
        self.min_len = cfg.min_len
        # GFN stuff
        self.train_steps = cfg.train_steps
        self.random_action_prob = cfg.random_action_prob
        self.batch_size = cfg.batch_size
        self.reward_min = cfg.reward_min
        self.gen_clip = cfg.gen_clip
        self.sampling_temp = cfg.sampling_temp
        self.sample_beta = cfg.sample_beta
        self.val_batch_size = cfg.val_batch_size
        self.eval_batch_size = cfg.eval_batch_size
        self.eval_samples = cfg.eval_samples
        # Eval Stuff
        self.eval_freq = cfg.eval_freq
        self.offline_gamma = cfg.offline_gamma
        self.eos_char = "[SEP]"
        self.pad_tok = self.tokenizer.convert_token_to_id("[PAD]")
        self.use_boltzmann = cfg.use_boltzmann

    def init_policy(self):
        cfg = self.cfg
        self.model = hydra.utils.instantiate(cfg.model)

        self.model.to(self.device)
        self.opt = torch.optim.Adam(self.model.model_params(), cfg.pi_lr, weight_decay=cfg.wd,
                            betas=(0.9, 0.999))
        self.opt_Z = torch.optim.Adam(self.model.Z_param(), cfg.z_lr, weight_decay=cfg.wd,
                            betas=(0.9, 0.999))

    def optimize(self, task, init_data=None, val_data=None):
        """
        optimize the task involving multiple objectives (all to be maximized) with 
        optional data to start with
        """
        # added to save model
        cfg = self.cfg
        losses, rewards, lens = [], [], []
        val_losses, rews = 0, 0
        pb = tqdm(range(self.train_steps))
        desc_str = "Evaluation := Reward: {:.3f} Val Loss: {:.3f} | Train := Loss: {:.3f} Rewards: {:.3f}"
        pb.set_description(desc_str.format(0, 0, sum(losses[-10:]) / 10, sum(rewards[-10:]) / 10 ))

        for i in pb:
            loss, r, le = self.train_step(task, self.batch_size, init_data)
            #experiment.log_metrics({"train_loss": loss, "train_reward": r}, step=i)
            losses.append(loss)
            rewards.append(r)
            lens.append(le)
            if i % self.eval_freq == 0:
                with torch.no_grad():
                    #rews, val_losses, eval_data = self.evaluation(task, val_data)
                    #experiment.log_metrics({"val_loss": val_losses, "eval_reward": rews}, step=i)
                    #figure = get_distribution_plot(eval_data["rewards"])
                    #experiment.log_figure("generated_samples", figure, step=i)
                    #plt.close(figure)
                    torch.save(self.model.state_dict(), "model_3_spins.pt") # save the model
                    samples, scores = self.generate(cfg.eval_batch_size, task)
                    np.save("samples_3_spins.npy", samples)
                    np.save("scores_3_spins.npy", scores)
            pb.set_description(desc_str.format(np.max(scores), 0, sum(losses[-10:]) / 10, sum(rewards[-10:]) / 10))
        
        return {
            'losses': losses,
            'train_rs': rewards
        }
    
    def sample_offline_data(self, dataset, batch_size):
        w = np.array(dataset[1])
        return np.random.choice(dataset[0], size=batch_size, replace=True, p = np.exp(w - w.max()) / np.exp(w-w.max()).sum(0))

    def train_step(self, task, batch_size, init_data=None):
        states, logprobs, lens = self.sample(batch_size)
        
        if init_data is not None and self.offline_gamma > 0 and int(self.offline_gamma * batch_size) > 0:
            offline_batch = self.sample_offline_data(init_data, int(self.offline_gamma * batch_size))
            offline_logprobs = self._get_log_prob(offline_batch)
            logprobs = torch.cat((logprobs, offline_logprobs), axis=0)
            states = np.concatenate((states, offline_batch), axis=0)
    
        r = self.process_reward(states.tolist(), task).to(self.device)
        self.opt.zero_grad()
        self.opt_Z.zero_grad()
        # TB Loss
        if self.use_boltzmann:
            loss = (logprobs - self.sample_beta * r).pow(2).mean()
        else:
            loss = (logprobs - self.sample_beta * r.clamp(self.reward_min).log()).pow(2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gen_clip)
        self.opt.step()
        self.opt_Z.step()
        return loss.item(), r.mean(), lens.float().mean()


    def sample(self, episodes, train=True):
        states = [''] * episodes
        traj_logprob = torch.zeros(episodes).to(self.device)

        active_mask = torch.ones(episodes).bool().to(self.device)
        x = str_to_tokens(states, self.tokenizer, use_sep=False).to(self.device).t()[:1].long()
        lens = torch.zeros(episodes).long().to(self.device)
        uniform_pol = torch.empty(episodes).fill_(self.random_action_prob).to(self.device)
        
        ep_len = self.max_len
        for t in (range(ep_len) if episodes > 0 else []):
            logits = self.model(x, lens=lens, mask=None)
            
            if t <= self.min_len:
                logits[:, 0] = -1000 # Prevent model from stopping
                                     # without having output anything
                if t == 0:
                    traj_logprob += self.model.Z()
            sampling_dist = Categorical(logits=logits / self.sampling_temp)
            pf_dist = Categorical(logits=logits)
            actions = sampling_dist.sample()
            if train and self.random_action_prob > 0:
                uniform_mix = torch.bernoulli(uniform_pol).bool()
                actions = torch.where(uniform_mix, torch.randint(int(t <= self.min_len), logits.shape[1], (episodes, )).to(self.device), actions)
            
            log_prob = pf_dist.log_prob(actions) * active_mask
            lens += torch.where(active_mask, torch.ones_like(lens), torch.zeros_like(lens))
            traj_logprob += log_prob

            actions_apply = torch.where(torch.logical_not(active_mask), torch.zeros(episodes).to(self.device).long(), actions + 4)
            active_mask = torch.where(active_mask, actions != 0, active_mask)

            x = torch.cat((x, actions_apply.unsqueeze(0)), axis=0)
            if active_mask.sum() == 0:
                break
        states = tokens_to_str(x.t(), self.tokenizer)
        return states, traj_logprob, lens
    
    def generate(self, num_samples, task):
        generated_seqs = []
        rewards = []
        while len(generated_seqs) < num_samples:
            with torch.no_grad():
                samples, _, l = self.sample(self.eval_batch_size, train=False)
                r = self.process_reward(samples.tolist(), task).cpu().numpy().tolist()
            generated_seqs.extend(samples.tolist())
            rewards.extend(r)
        return np.array(generated_seqs), np.array(rewards)

    def process_reward(self, seqs, task):
        return torch.tensor(task(seqs))

    def val_step(self, val_data, batch_size, task):
        overall_loss = 0.
        num_batches = max(1, len(val_data[0]) // batch_size)
        losses = 0
        for i in range(num_batches):
            states = val_data[0][i*batch_size:(i+1)*batch_size]
            logprobs = self._get_log_prob(states)

            r = self.process_reward(states, task).to(logprobs.device)
            loss = (logprobs - self.sample_beta * r.clamp(min=self.reward_min).log()).pow(2).mean()

            losses += loss.item()
        overall_loss += (losses)
        return overall_loss / num_batches

    def evaluation(self, task, val_data):
        val_loss = self.val_step(val_data, self.val_batch_size, task)
        samples, rewards = self.generate(self.eval_samples, task)
        return rewards.mean(), val_loss, {'samples': samples, 'rewards': rewards.tolist()}

    def _get_log_prob(self, states):
        lens = torch.tensor([len(z) + 2 for z in states]).long().to(self.device)
        x = str_to_tokens(states, self.tokenizer).to(self.device).t()
        mask = x.eq(self.tokenizer.padding_idx)
        logits = self.model(x, mask=mask.transpose(1,0), return_all=True, lens=lens, logsoftmax=True)
        seq_logits = (logits.reshape(-1, 4)[torch.arange(x.shape[0] * x.shape[1], device=self.device), (x.reshape(-1)-5).clamp(0)].reshape(x.shape) * mask.logical_not().float()).sum(0)
        seq_logits += self.model.Z()
        return seq_logits


class PottsReward:
    def __init__(self, num_of_elements, tokenizer, prefix="/home/mila/m/moksh.jain/BioSeq-GFN-AL/data/"):
        """
        load model and tokenizer
        """
        self.tokenizer = tokenizer
        self.J=np.zeros(shape=(4,4,num_of_elements))
        for ii in range(num_of_elements):
            nextJJ=prefix+'potts/JJ_'+str(ii+1)+'.mat'
            Jdict=sco.loadmat(nextJJ)
            self.J[:,:,ii]=Jdict['JJ_out']

        hdict=sco.loadmat(prefix+'potts/h_out.mat')
        self.h=hdict['h']
    
    def __call__(self, seqs):
        seqs = str_to_tokens(seqs, self.tokenizer).numpy() - 4
        seqs = seqs[:, 1:-1]

        # introduce new reward function for spin dynamics problem
        
        # load in reference population from full dynamics, along woth initial condition combos
        # and arrays for influence functionals S0, S, Sb, Sp for corresponding system
        reference_pop = np.load(
        "full_three_spins_coherent_reference_pop.npy")
        initial_combos = np.load(
        "zero_one_two_blips_three_spins_coherent_initial_combos.npy")
        S0 = np.load("three_spins_coherent_S0.npy")
        S = np.load("three_spins_coherent_S.npy")
        Sb = np.load("three_spins_coherent_Sb.npy")
        Sp = np.load("three_spins_coherent_Sp.npy")

        #reference_pop = np.load(
        #"full_three_spins_reference_pop.npy")
        #initial_combos = np.load(
        #"zero_one_blips_three_spins_initial_combos.npy")
        #S0 = np.load("three_spins_S0.npy")
        #S = np.load("three_spins_S.npy")
        #Sb = np.load("three_spins_Sb.npy")
        #Sp = np.load("three_spins_Sp.npy")

        # function to convert GFN representation sequence to spin representation combo
        def GFNseq_to_spinrep(sequence, Ns):
            num_combo = int(len(sequence)/Ns)
            #print(num_combo)
            GFN_rep_combos = np.reshape(sequence, (num_combo, Ns))
            #print(GFN_rep_combos)
            spin_rep_combo = np.zeros((num_combo, 2*Ns), dtype="int")
            for index1 in range(num_combo):
                for index2 in range(Ns):
                    if GFN_rep_combos[index1, index2] == 1:  # [0, 0]
                        spin_rep_combo[index1, index2] = 0
                        spin_rep_combo[index1, index2+Ns] = 0
                    if GFN_rep_combos[index1, index2] == 2:  # [1, 0]
                        spin_rep_combo[index1, index2] = 1
                        spin_rep_combo[index1, index2+Ns] = 0
                    if GFN_rep_combos[index1, index2] == 3:  # [0, 1]
                        spin_rep_combo[index1, index2] = 0
                        spin_rep_combo[index1, index2+Ns] = 1
                    if GFN_rep_combos[index1, index2] == 4:  # [1, 1]
                        spin_rep_combo[index1, index2] = 1
                        spin_rep_combo[index1, index2+Ns] = 1
            return spin_rep_combo
        
        Ns = 3  # number of spins
        Nlev = 2  # levels to the system
        Nt = Ns*2  # deg
        num = Nlev**Nt
        it = Nt-2

        # function to convert index to spin representation combo
        def index_to_combo(index):
            combo = np.zeros(Nt, dtype="int")
            remainder = 2
            num_loops = Nt

            for iteration in range(num_loops):
                value = int(index%remainder>=remainder/2)
                if iteration%2==0:
                    combo[Nt-1-(iteration//2)] = value
                else:
                    combo[int(Nt/2)-1-(iteration//2)] = value
                remainder = 2*remainder

            return combo

        # function to convert spin representation combo to index
        def combo_to_index(combo):
            one_indices = np.where((combo == 1)==True)[0]
            powers = np.zeros(Nt, dtype="int")
            num_loops = Nt

            for iteration in range(num_loops):
                value = 2**iteration
                if iteration%2==0:
                    powers[Nt-1-(iteration//2)] = value
                else:
                    powers[int(Nt/2)-1-(iteration//2)] = value

            index = np.sum(powers[one_indices])
            return index

        # function to insert spin representation combos into an array of spin representation combos in proper order
        # according to indices
        def insert_combos(combos_to_insert, list_of_combos):
            list_of_indices = np.zeros(len(list_of_combos), dtype="int")
            for counter1 in range(len(list_of_combos)):
                index = combo_to_index(list_of_combos[counter1])
                list_of_indices[counter1] = index
            indices_to_insert = np.zeros(len(combos_to_insert), dtype="int")
            for counter2 in range(len(combos_to_insert)):
                index = combo_to_index(combos_to_insert[counter2])
                indices_to_insert[counter2] = index
            indices_insert_doubles_removed = indices_to_insert
            values_deleted = np.array([], dtype="int")
            for counter3 in range(len(indices_to_insert)):
                where_vals = np.array(np.where(indices_insert_doubles_removed == indices_to_insert[counter3]))[0]
                # only want one entry of a given index in the list
                if len(where_vals) > 1 and indices_to_insert[counter3] not in values_deleted:
                    indices_to_delete = where_vals[1:]
                    indices_insert_doubles_removed = np.delete(indices_insert_doubles_removed, indices_to_delete)
                    values_deleted = np.append(values_deleted, indices_to_insert[counter3])
            new_indices_to_insert = indices_insert_doubles_removed
            for counter4 in range(len(indices_insert_doubles_removed)):
                # remove copies so that two of the same spin pathway are not used in the dynamics
                if indices_insert_doubles_removed[counter4] in list_of_indices:
                    new_indices_to_insert = np.delete(new_indices_to_insert, np.array(
                        np.where(new_indices_to_insert == indices_insert_doubles_removed[counter4]))[0])
            counter5 = 0
            updated_list_of_indices = list_of_indices
            for index_to_insert in new_indices_to_insert:
                for index in updated_list_of_indices:
                    if index < index_to_insert:
                        counter5 += 1
                updated_list_of_indices = np.insert(updated_list_of_indices, counter5, index_to_insert)
                counter5 = 0
            updated_list_of_combos = np.zeros((len(updated_list_of_indices), Nt), dtype="int")
            for counter6 in range(len(updated_list_of_indices)):
                index = updated_list_of_indices[counter6]
                combo = index_to_combo(index)
                updated_list_of_combos[counter6] = combo
            return updated_list_of_combos, new_indices_to_insert

        #def insert_combos(combos_to_insert, list_of_combos):
            #list_of_indices = np.zeros(len(list_of_combos), dtype="int")
            #for counter1 in range(len(list_of_combos)):
                #index = combo_to_index(list_of_combos[counter1])
                #list_of_indices[counter1] = index
            #indices_to_insert = np.zeros(len(combos_to_insert), dtype="int")
            #for counter2 in range(len(combos_to_insert)):
                #index = combo_to_index(combos_to_insert[counter2])
                #indices_to_insert[counter2] = index
            #new_indices_to_insert = indices_to_insert
            #for counter3 in range(len(indices_to_insert)):
                #if indices_to_insert[counter3] in list_of_indices: # do not want doubles here, need to remove from indices to insert
                    #new_indices_to_insert = np.delete(indices_to_insert, counter3)
            #counter4 = 0
            #updated_list_of_indices = list_of_indices
            #for index_to_insert in new_indices_to_insert:
                #for index in updated_list_of_indices:
                    #if index < index_to_insert:
                        #counter4 += 1
                #updated_list_of_indices = np.insert(updated_list_of_indices, counter4, index_to_insert)
                #counter4 = 0
            #print(updated_list_of_indices)
            #updated_list_of_combos = np.zeros((len(updated_list_of_indices), Nt), dtype="int")
            #for counter5 in range(len(updated_list_of_indices)):
                #index = updated_list_of_indices[counter5]
                #combo = index_to_combo(index)
                #updated_list_of_combos[counter5] = combo
            #return updated_list_of_combos
        
        # indices of spin combos used in dynamics need to be formatted properly
        # this function will return the appropriately formatted objects, filled with indices of combos used in dynamics
        def get_summation_indices(indices_for_dynamics):
            ordered_combo_indices = np.arange(num, dtype="int")
            IND = np.empty(0, dtype="int")
            aa = np.full((Nlev**it, Nlev**2), -1, dtype="int")
            b = np.full((Nlev**2, Nlev**it), -1, dtype="int")
            bi = 0
            quart = 1
            first = np.full(Nlev**it, -1, dtype="int")
            row = 0
            counter = 0

            for index in ordered_combo_indices:
                if index in indices_for_dynamics:
                    IND = np.append(IND, counter)
                    if (counter+1)%(Nlev**(Nt-2)) != 0:
                        layera = counter%(Nlev**(Nt-2))
                    else:
                        layera = Nlev**(Nt-2)-1
                    if (counter+1)%(Nlev**(Nt-2)) != 0:  # this if is never true, skips to else
                        if quart < np.floor((counter+1)/Nlev**(Nt-2))+1:
                            index = np.where(b[bi] == -1)[0][0]  # first index where b is -1, will replace this value
                            b[bi] = first
                            bi += 1
                            first = np.full(Nlev**it, -1, dtype="int")
                            index = np.where(first == -1)[0][0]  # first index where first is -1, will replace this value
                            first[index] = row
                            quart = np.floor(counter/Nlev**(Nt-2))+1
                        else:
                            index = np.where(first == -1)[0][0]  # first index where first is -1, will replace this value
                            first[index] = row
                    else:  # (counter+1)%(Nlev**(Nt-2)) = 0
                        if quart == np.floor((counter+1)/Nlev**(Nt-2)):
                            index = np.where(first == -1)[0][0]  # first index where first is -1, will replace this value
                            first[index] = row
                            index = np.where(b[bi] == -1)[0][0]  # first index where b is -1, will replace this value
                            b[bi] = first
                            bi += 1
                            first = np.full(Nlev**it, -1, dtype="int")
                            quart = np.floor((counter+1)/Nlev**(Nt-2))+1
                        # else:
                            # b[bi] = np.append(b[bi], first)
                            # bi += 1
                            # b[bi] = np.append(b[bi], row)
                            # bi += 1
                            # quart = np.floor(counter/Nlev**(Nt-2))

                    index = np.where(aa[layera] == -1)[0][0]  # first index where aa is -1, will replace this value
                    aa[layera, index] = row
                    row = row + 1

                counter = counter + 1

            return IND, aa, b
        
        # function to compute the dynamics over a certain number of iterations, and return the population and trace arrays
        def compute_dynamics(IND, aa, b, S0, S, Sb, Sp):
            # num = Nlev**Nt
            dim = int(num/Nlev**2)
            tot_iter = 500 #4000

            coefs = np.zeros(Nlev**2, dtype="complex")
            coefs[0] = 1 # initial condition
            pop = np.zeros((tot_iter+1, Nlev**2), dtype="complex")
            trace = np.zeros(tot_iter+1, dtype="complex")

            for index_coefs in range(Nlev**2):
                pop[0, index_coefs] = coefs[index_coefs]

            trace[0] = np.trace(pop[0].reshape(Nlev, Nlev))

            rho = np.zeros(dim, dtype="complex")
            indices = b[0]
            selected_indices = indices[indices >= 0] # neglect indices that are -1 in b
            rho[IND[selected_indices]] = S0[selected_indices]

            bin_ranges = np.arange(0, num+1+Nlev**2, Nlev**2)

            Nlsp = np.histogram(IND, bin_ranges)[0]
            numls = Nlsp[0:len(Nlsp)-1]

            S0 = S0[IND] # remove all indices from the IFs not used in the dynamics
            S = S[IND]
            Sb = Sb[IND]
            Sp = Sp[IND]

            for iter in range(1, tot_iter+1):
                rhop = rho
                rhoe = np.array([])

                # expand
                for kk in range(len(rho)):
                    if kk == 0:
                        rhoe = np.array(np.kron(np.ones(numls[kk]), rhop[kk]))
                    else:
                        rhoe = np.append(rhoe, np.array(np.kron(np.ones(numls[kk]), rhop[kk])))

                rho2 = rhoe*S/Sp

                # contract
                for kk in range(len(aa)):
                    indices = aa[kk]
                    selected_indices = indices[indices > -1]  # neglect indices that are -1 in aa
                    rho[kk] = np.sum(rho2[selected_indices])

                rho2end = rhoe*Sb/Sp
                rhoend = np.zeros(dim, dtype="complex")

                for kk in range(len(aa)):
                    indices = aa[kk]
                    selected_indices = indices[indices > -1] # neglect indices that are -1 in aa
                    rhoend[kk] = np.sum(rho2end[selected_indices])

                rhoendc = rhoend

                indices = np.zeros((Nlev**2, int(len(aa)/Nlev**2)), dtype="int")
                for index1 in range(Nlev**2):
                    for index2 in range(int(len(aa)/Nlev**2)):
                        indices[index1, index2] = index2*Nlev**2+index1

                for kk in range(Nlev**2):
                    pop[iter, kk] = np.sum(rhoendc[indices[kk]])

                trace[iter] = np.trace(pop[iter].reshape(Nlev, Nlev))

            return pop, trace
        
        # count number of blips in a spin combo
        def count_blips(combo):
            num_bl = 0
            for index in range(Ns):
                if combo[index] != combo[index+Ns]:
                    num_bl += 1
            return num_bl

        # function to compute the final reward coming directly from the populations, trace of the dynamics, and combos added by GFN
        def compute_reward(pop, trace, indices_added):
            scaling_factor1 = 50 # scaling factor serving as a prefactor for first exponential reward
            scaling_factor2 = 5 # scaling factor serving as temperature for the exponential rewards
            scaling_factor3 = 25 # scaling factor serving as a prefactor for second and third exponential reward
            
            # sum over all iterations to favour correct behaviour over all dynamics
            # first part of reward - conserved and consistent trace (close to 1, small changes over all iterations)
            reward1 = scaling_factor1*np.exp(-np.absolute(np.sum(np.real(trace)-1))/scaling_factor2)
            # part 2 of first reward: consistent trace (small changes between iterations)
            #reward1_2 = scaling_factor1*np.exp(-np.absolute(np.sum(np.real(trace[2:len(trace)]-trace[1:len(trace)-1])))/scaling_factor2)
            # first part of reward - conserved trace close to 1
            #reward1_1 = np.sum(-np.log10(1-np.real(trace[1:])))
            #if np.isnan(reward1_1) == True or np.isinf(reward1_1) == True:
                #reward1_1 = 0 #20000
            
            # second part of reward - multiple spin pathways with 2 blips
            #scale_factor = 2500
            #reward1_2 = scale_factor*len(indices_added)
            #num_blips = np.array([count_blips(index_to_combo(index)) for index in indices_added])
            #where_2_blips = np.array(np.where(num_blips == 2)[0])
            #reward1_2 = scale_factor*len(where_2_blips)

            #reward1=reward1_2+reward1_1

            # second part of reward - real populations: favour small imaginary parts of populations
            reward2 = scaling_factor3*(np.exp(-np.absolute(np.sum(np.imag(pop[:, 0])))/scaling_factor2)+
            np.exp(-np.absolute(np.sum(np.imag(pop[:, 3])))/scaling_factor2))
            
            # third part of reward - similar populations to reference: favour small changes in physics
            reward3 = scaling_factor3*(np.exp(-np.absolute(np.sum(
            np.real(pop[:, 0]-reference_pop[:501, 0])))/scaling_factor2)+
            np.exp(-np.absolute(np.sum(np.real(pop[:, 3]-reference_pop[:501, 3])))/scaling_factor2))

            #print("Reward 1 Part 1: "+str(reward1_1))
            #print("Reward 1 Part 2: "+str(reward1_2))
            #print("Reward 1: "+str(reward1))
            #print("Reward 2: "+str(reward2))
            #print("Reward 3: "+str(reward3))

            reward = reward1+reward2+reward3

            return reward
        
        rewards = np.array([])
        for seq in seqs:
            # combos outputted by GFN in spin representation (0, 1) instead of GFN representation (1, 2, 3, 4)
            spin_combos = GFNseq_to_spinrep(seq, Ns)

            # array of combined combos from reference and GFN, along with indices of new combos
            reference_and_GFN_combos, new_indices_added = insert_combos(spin_combos, initial_combos)
            reference_and_GFN_indices = np.array([combo_to_index(combo) for combo in reference_and_GFN_combos])
            # now, need to run dynamics on these spin combos to get populations and trace in order to compute final reward
            # first, get properly formatted indices for summation during dynamics

            IND, aa, b = get_summation_indices(reference_and_GFN_indices) # summation indices formatted properly for dynamics
            
            # trace and populations from dynamics
            pop, trace = compute_dynamics(IND, aa, b, S0, S, Sb, Sp)

            # use trace, populations, and added combos from GFN to compute final reward
            reward = compute_reward(pop, trace, new_indices_added)
            rewards = np.append(rewards, reward)

        return rewards

        #h = self.h
        #J = self.J
        #energies = []
        #for seq in seqs:
            #N=np.size(seq)
            #seq=seq-1
            #for ii in range(N):
                #energy=0;
                #l=0;
                #for i in range(N-1):
                    #for j in range(i+1, N):
                        #energy=energy+J[seq[i],seq[j],l];
                        #l=l+1;
            #for i in range(N):
                    #energy=energy+h[seq[i],i];
            #energies.append(energy)
        #return np.array(energies)


@hydra.main(config_path='./config', config_name='potts.yaml')
def main(config):
    random.seed(None)

    log_config = flatten_config(OmegaConf.to_container(config, resolve=True), sep='/')
    log_config = {'/'.join(('config', key)): val for key, val in log_config.items()}
    
    # config['job_name'] = wandb.run.name
    config = init_run(config)
    
    #experiment.log_parameters(log_config)

    tokenizer = hydra.utils.instantiate(config.tokenizer)
    dataset = hydra.utils.instantiate(config.dataset, tokenizer=tokenizer)

    task = PottsReward(config.num_of_elements, tokenizer, prefix=config.prefix)

    generator = GFN(config.gfn, tokenizer)
    #generator.model.load_state_dict(torch.load("model_1epoch_3spins.pt")) # load the model
    generator.optimize(task, init_data=dataset.training_data, val_data=None) #dataset.validation_set())
    torch.save(generator.model.state_dict(), "model_3_spins.pt") # save the model

    samples, scores = generator.generate(config.num_samples, task)
    np.save("samples_3_spins.npy", samples)
    np.save("scores_3_spins.npy", scores)
    figure = get_distribution_plot(scores)
    #experiment.log_figure(figure_name="final_distribution", figure=figure)
    plt.close(figure)
    #experiment.log_others({'samples': samples, 'scores': scores.tolist()})

if __name__ == "__main__":
    #experiment = Experiment(project_name="GFN-Potts")
    main()
