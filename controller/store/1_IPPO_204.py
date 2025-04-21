import os
import numpy as np
import torch
import torch.nn as nn
import time
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from controller.ppo.actor.UnetActor import UNet
from controller.ppo.critic.CNNCritic import CNNCritic
import copy
import shutil
import csv
from rl_env.WRSN import WRSN
import threading
import time
import threading
import queue
import copy

import csv
import multiprocessing as mp
from torch.multiprocessing import Queue, Process, set_start_method


class IPPO:
    def __init__(self, args, env, device, model_path=None):
        self.env = env
        self.num_agent = env.num_agent
        self.model_path = model_path
        self.device = device
        self.gamma = args["gamma"]
        self.clip = args["clip"]
        self.batch_size = args["batch_size"]
        self.minibatch_size = args["minibatch_size"]
        self.n_updates_per_iteration = args["n_updates_per_iteration"]
        self.save_freq = args["save_freq"]
        self.gae = args["gae"]
        self.clip_vloss = args["clip_vloss"]
        self.ent_coef = args["ent_coef"]
        self.vf_coef = args["vf_coef"]
        self.gae_lambda = args["gae_lambda"]
        self.norm_adv = args["norm_adv"]
        self.max_grad_norm = args["max_grad_norm"]
        self.actors = [UNet().to(self.device) for _ in range(self.num_agent)]
        self.critics = [CNNCritic().to(self.device) for _ in range(self.num_agent)]
        self.log_file = [None for _ in range(self.num_agent)]

        self.loggers = [{
            'i_so_far': 0,          # iterations so far
			't_so_far': 0,          # timesteps so far
			'ep_lens': [],       # episodic lengths in batch
			'ep_lifetime': [],       # episodic returns in batch
			'losses': [],     # losses of actor network in current iteration
            'rewards': [],
            'delta_t': time.time_ns()
		} for _ in range(self.num_agent)]

        if model_path is not None:
            agent_folders = os.listdir(model_path)
            for agent_folder in agent_folders:
                agent_index = int(agent_folder)
                agent_path = os.path.join(model_path, agent_folder)
                self.critics[agent_index].load_state_dict(torch.load(os.path.join(agent_path, "critic.pth")))
                self.actors[agent_index].load_state_dict(torch.load(os.path.join(agent_path, "actor.pth")))   
                self.critics[agent_index].to(self.device)
                self.actors[agent_index].to(self.device)
                self.log_file[agent_index] = os.path.join(agent_path, "log.csv")
                with open(self.log_file[agent_index], 'r') as file:
                    reader = csv.reader(file)
                    rows = list(reader)
                    last_row = rows[-1] if rows else []
                    self.loggers[agent_index]['t_so_far'] = int(last_row[1])
                    self.loggers[agent_index]['i_so_far'] = int(last_row[0])
        self.optimizers = [None for _ in range(self.num_agent)]
        for i in range(self.num_agent):
            parameters = list(self.actors[i].parameters()) + list(self.critics[i].parameters())
            # Create optimizer for combined parameters
            self.optimizers[i] = optim.Adam(parameters, lr=args["lr"])

    def cal_rt_adv(self, id, states, rewards, next_states, terminals):
        with torch.no_grad():
            values = self.get_value(id, states)
            next_values = self.get_value(id, next_states)

            if self.gae:
                advantages = torch.zeros_like(rewards)
                lastgaelam = 0
                for t in reversed(range(len(rewards))): 
                    delta = rewards[t] + self.gamma * next_values[t] * int(terminals[t]) - values[t]
                    lastgaelam = delta + self.gamma * self.gae_lambda * int(terminals[t]) * lastgaelam
                    advantages[t] = lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards)
                for t in reversed(range(len(rewards))):
                    if t == len(rewards):
                        next_return = next_values[t]
                    else:
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + self.gamma * terminals[t] * next_return
                advantages = returns - values
            return returns, advantages, values
    
    def get_action(self, agent_id, state_in):
        # state_in = np.zeros((4, 100, 100))
        # # Kiểm tra dữ liệu
        print("Shape:", state_in.shape)
        print("Type:", state_in.dtype)
        np_state = np.array(state_in, dtype=np.float32)
        state = torch.from_numpy(np_state)
        print(state)

        print("2")
        if state.ndim == 3:
            state = torch.unsqueeze(state, dim=0)
        print("3")

        mean, log_std = self.actors[agent_id](state)
        std = log_std.exp()
        dist = Normal(mean, std)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)

        return action.detach().cpu().numpy(), action_log_prob.sum().detach().cpu().numpy()
    
    def evaluate(self, agent_id, batch_states, batch_actions):
        mean, log_std = self.actors[agent_id](batch_states)
        std = torch.exp(log_std)
        
        dist = Normal(mean, std)
        action_log_prob = dist.log_prob(batch_actions)
        return action_log_prob.sum((1, 2)), dist.entropy().sum((1,2))

    def get_value(self, agent_id, state):
        value = self.critics[agent_id](state)
        return value.sum(1)
    

        # Hàm khởi tạo môi trường riêng
    
    
    def create_environment():
        return WRSN(scenario_path="physical_env/network/network_scenarios/hanoi1000n50.yaml",
                   agent_type_path="physical_env/mc/mc_types/default.yaml",
                   num_agent=3, map_size=100, density_map=True)
    
    def roll_out(self):
        batch_states = [[] for _ in range(self.num_agent)]
        batch_actions = [[] for _ in range(self.num_agent)]
        batch_log_probs = [[] for _ in range(self.num_agent)]
        batch_next_states = [[] for _ in range(self.num_agent)]
        batch_rewards = [[] for _ in range(self.num_agent)]
        batch_returns = [[] for _ in range(self.num_agent)]
        batch_advantages = [[] for _ in range(self.num_agent)]
        batch_values = [[] for _ in range(self.num_agent)]
        
        t = [0 for _ in range(self.num_agent)]
        while True:
            states = [[] for _ in range(self.num_agent)]
            actions = [[] for _ in range(self.num_agent)]
            log_probs = [[] for _ in range(self.num_agent)]
            next_states = [[] for _ in range(self.num_agent)]
            rewards = [[] for _ in range(self.num_agent)]
            terminals = [[] for _ in range(self.num_agent)]
            request = self.env.reset()
            cnt = [0 for _ in range(self.num_agent)]
            log_probs_pre = [None for _ in range(self.num_agent)]
            while True:
                action, log_prob = self.get_action(request["agent_id"], request["state"])
                log_probs_pre[request["agent_id"]] = log_prob
                request = self.env.step(request["agent_id"], action)
                if request["terminal"]:
                    break
                if log_probs_pre[request["agent_id"]] is None:
                    continue
                t[request["agent_id"]] += 1
                cnt[request["agent_id"]] += 1
                states[request["agent_id"]].append(request["prev_state"])
                actions[request["agent_id"]].append(request["input_action"])
                next_states[request["agent_id"]].append(request["state"])
                rewards[request["agent_id"]].append(request["reward"])
                log_probs[request["agent_id"]].append(log_probs_pre[request["agent_id"]])
                terminals[request["agent_id"]].append(request["terminal"])

                self.writer_log_all.writerow([
                    request["agent_id"],
                    round(request["action"][0], 4),
                    round(request["action"][1], 4),
                    round(request["action"][2], 4),
                    round(request["detailed_rewards"][0], 4),
                    round(request["detailed_rewards"][1], 4),
                    round(request["detailed_rewards"][2], 4)
                ])
                self.file_log_all.flush()



            for id in range(self.num_agent):
                if len(states[id]) == 0:
                    continue
                returns, advantages, values = self.cal_rt_adv(id, torch.Tensor(np.array(states[id])).to(self.device), torch.Tensor(np.array(rewards[id])).to(self.device), torch.Tensor(np.array(next_states[id])).to(self.device), torch.Tensor(np.array(terminals[id])).to(self.device))
                batch_states[id].extend(states[id])
                batch_actions[id].extend(actions[id])
                batch_log_probs[id].extend(log_probs[id])
                batch_next_states[id].extend(next_states[id])
                batch_rewards[id].extend(rewards[id])
                batch_advantages[id].extend(advantages)
                batch_returns[id].extend(returns)
                batch_values[id].extend(values)

            for id in range(self.num_agent):
                self.loggers[id]["ep_lens"].append(cnt)
                self.loggers[id]["ep_lifetime"].append(self.env.env.now)
                self.loggers[id]["rewards"].append(np.array(rewards[id]))
            out = True
            for element in t:
                if element < self.batch_size:
                    out = False
                    break
            if out:
                break

        for id in range(self.num_agent):
            mean = np.mean(batch_rewards[id])
            # Calculate the absolute differences between elements and the 50th percentile
            abs_diff = np.abs(batch_rewards[id] - mean)
            indices = np.argsort(abs_diff)
            selected_num = int(self.batch_size / 2.0)
            random_num = self.batch_size - selected_num
            indices = np.concatenate((indices[-selected_num:], np.random.choice(len(batch_rewards[id]) - selected_num, size=random_num, replace=False)))

            batch_rewards[id] = torch.FloatTensor(np.array(batch_rewards[id])[indices]).to(self.device)
            batch_states[id] = torch.FloatTensor(np.array(batch_states[id])[indices]).to(self.device)
            batch_actions[id] = torch.FloatTensor(np.array(batch_actions[id])[indices]).to(self.device)
            batch_next_states[id] = torch.FloatTensor(np.array(batch_next_states[id])[indices]).to(self.device)
            batch_log_probs[id] = torch.FloatTensor(np.array(batch_log_probs[id])[indices]).to(self.device)
            batch_returns[id] = torch.stack([batch_returns[id][_] for _ in indices]).to(self.device)
            batch_advantages[id] = torch.stack([batch_advantages[id][_] for _ in indices]).to(self.device)
            batch_values[id] = torch.stack([batch_values[id][_] for _ in indices]).to(self.device)
        return batch_states, batch_actions, batch_log_probs, batch_rewards, batch_next_states, batch_advantages, batch_returns, batch_values
   
    # def train(self, trained_iterations, save_folder):
    #     self.file_log_all = open('log_cur.csv', 'w', newline='')
    #     self.writer_log_all = csv.writer(self.file_log_all)
    #     writers = [SummaryWriter(f"runs/ippo/{qq}") for qq in range(self.num_agent)]
    #     start_time = time.time()
    #     t_so_far = 0 
    #     i_so_far = 0 
    #     logs = [[] for _ in range(self.num_agent)]
    #     while i_so_far <= trained_iterations:                                           
    #         # batch_states_all, batch_actions_all, batch_log_probs_all, batch_rewards_all, batch_next_states_all, batch_advantages_all, batch_returns_all, batch_values_all = self.roll_out()
    #         batch_states_all, batch_actions_all, batch_log_probs_all, batch_rewards_all, batch_next_states_all, batch_advantages_all, batch_returns_all, batch_values_all = self.parallel_roll_out(num_threads=4)
    #         t_so_far += self.batch_size
    #         i_so_far += 1
            
    #         for id in range(self.num_agent):
    #             self.loggers[id]['t_so_far'] += self.batch_size
    #             self.loggers[id]['i_so_far'] += 1
    #             batch_states, batch_actions, batch_log_probs, batch_rewards, batch_next_states, batch_advantages, batch_returns, batch_values = batch_states_all[id], batch_actions_all[id], batch_log_probs_all[id], batch_rewards_all[id], batch_next_states_all[id], batch_advantages_all[id], batch_returns_all[id], batch_values_all[id]
    #             b_inds = np.arange(self.batch_size)
    #             clipfracs = []
    #             for _ in range(self.n_updates_per_iteration):
    #                 np.random.shuffle(b_inds)
    #                 for start in range(0, self.batch_size, self.minibatch_size):
    #                     end = start + self.minibatch_size
    #                     mb_inds = b_inds[start:end]
    #                     newlogprob, entropy = self.evaluate(id, batch_states[mb_inds], batch_actions[mb_inds])
    #                     newvalue = torch.squeeze(self.get_value(id, batch_states[mb_inds]))
    #                     logratio = newlogprob - batch_log_probs[mb_inds]
    #                     ratio = logratio.exp()
    #                     with torch.no_grad():
    #                         old_approx_kl = (-logratio).mean()
    #                         approx_kl = ((ratio - 1) - logratio).mean()
    #                         clipfracs += [((ratio - 1.0).abs() > self.clip).float().mean().item()]
    #                     mb_advantages = batch_advantages[mb_inds]
    #                     if self.norm_adv:
    #                         mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
    #                     pg_loss1 = -mb_advantages * ratio
    #                     pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip, 1 + self.clip)
    #                     pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    #                     newvalue = newvalue.view(-1)
    #                     if self.clip_vloss:
    #                         v_loss_unclipped = (newvalue - batch_returns[mb_inds]) ** 2
    #                         v_clipped = batch_values[mb_inds] + torch.clamp(
    #                             newvalue - batch_values[mb_inds],
    #                             -self.clip,
    #                             self.clip,
    #                         )
    #                         v_loss_clipped = (v_clipped - batch_returns[mb_inds]) ** 2
    #                         v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
    #                         v_loss = 0.5 * v_loss_max.mean()
    #                     else:
    #                         v_loss = 0.5 * ((newvalue - batch_returns[mb_inds]) ** 2).mean()
    #                     entropy_loss = entropy.mean()
    #                     loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef
    #                     self.optimizers[id].zero_grad()
    #                     loss.backward()
                        
    #                     nn.utils.clip_grad_norm_(self.actors[id].parameters(), self.max_grad_norm)
    #                     nn.utils.clip_grad_norm_(self.critics[id].parameters(), self.max_grad_norm)
    #                     self.optimizers[id].step()
                    

    #                     self.loggers[id]['losses'].append(torch.clone(loss).detach().cpu().numpy())
    #             y_pred, y_true = batch_values.detach().cpu().numpy(), batch_returns.detach().cpu().numpy()
    #             var_y = np.var(y_true)
    #             explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    #             # TRY NOT TO MODIFY: record rewards for plotting purposes
    #             writers[id].add_scalar("charts/learning_rate", self.optimizers[id].param_groups[0]["lr"], i_so_far)
    #             writers[id].add_scalar("losses/value_loss", v_loss.item(), i_so_far)
    #             writers[id].add_scalar("losses/policy_loss", pg_loss.item(), i_so_far)
    #             writers[id].add_scalar("losses/entropy", entropy_loss.item(), i_so_far)
    #             writers[id].add_scalar("losses/old_approx_kl", old_approx_kl.item(), i_so_far)
    #             writers[id].add_scalar("losses/approx_kl", approx_kl.item(), i_so_far)
    #             writers[id].add_scalar("losses/clipfrac", np.mean(clipfracs), i_so_far)
    #             writers[id].add_scalar("losses/explained_variance", explained_var, i_so_far)
    #             print("SPS:", int(i_so_far) / (time.time() - start_time))
    #             writers[id].add_scalar("charts/SPS", int(i_so_far) / (time.time() - start_time), i_so_far)
                    
    #             # Print a summary of our training so far
    #             logs[id].append(self._log_summary(id))
    #             # Open the file in append mode, creating it if it doesn't exist

    #             # Save our model if it's time
    #             if self.loggers[id]['i_so_far'] % self.save_freq == 0:
    #                 folder = os.path.join(save_folder, os.path.join(str(self.loggers[id]['i_so_far']), str(id)))
    #                 if not os.path.exists(folder):
    #                     os.makedirs(folder)
    #                 torch.save(self.actors[id].state_dict(), os.path.join(folder, 'actor.pth'))
    #                 torch.save(self.critics[id].state_dict(), os.path.join(folder, 'critic.pth'))
    #                 if self.log_file[id] is not None:
    #                     shutil.copy(self.log_file[id], folder)
    #                 with open(os.path.join(folder, 'log.csv'), 'a', newline='') as file:
    #                     writer_log = csv.writer(file)
    #                     for row in logs[id]:
    #                         writer_log.writerow(row)
    #                 logs[id] = []
    #                 self.log_file[id] = os.path.join(folder, 'log.csv')
    #     self.file_log_all.close()



    def train(self, trained_iterations, save_folder):
        # Kiểm tra và tạo thư mục checkpoint nếu chưa tồn tại
        checkpoint_dir = os.path.join(save_folder, 'checkpoints')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        # Đường dẫn đến checkpoint gần nhất
        last_checkpoint_path = os.path.join(checkpoint_dir, 'last_checkpoint.pt')
        
        # Biến để kiểm soát việc tiếp tục từ checkpoint
        start_iteration = 0
        t_so_far = 0
        logs = [[] for _ in range(self.num_agent)]
        
        # Kiểm tra xem có checkpoint gần nhất không và nạp nó
        if os.path.exists(last_checkpoint_path):
            try:
                print(f"Tìm thấy checkpoint gần nhất tại {last_checkpoint_path}. Đang nạp...")
                checkpoint = torch.load(last_checkpoint_path)
                
                # Nạp trạng thái mô hình và quá trình huấn luyện
                for id in range(self.num_agent):
                    self.actors[id].load_state_dict(checkpoint['actor_state_dicts'][id])
                    self.critics[id].load_state_dict(checkpoint['critic_state_dicts'][id])
                    self.optimizers[id].load_state_dict(checkpoint['optimizer_state_dicts'][id])
                    self.loggers[id] = checkpoint['loggers'][id]
                
                # Nạp các biến theo dõi tiến trình
                start_iteration = checkpoint['iteration']
                t_so_far = checkpoint['t_so_far']
                logs = checkpoint['logs']
                
                print(f"Nạp thành công, tiếp tục huấn luyện từ vòng lặp {start_iteration}")
            except Exception as e:
                print(f"Lỗi khi nạp checkpoint: {e}")
                print("Khởi động lại từ đầu...")
                start_iteration = 0
                t_so_far = 0
                logs = [[] for _ in range(self.num_agent)]
        
        # Thiết lập logging
        self.file_log_all = open('log_cur.csv', 'w', newline='')
        self.writer_log_all = csv.writer(self.file_log_all)
        writers = [SummaryWriter(f"runs/ippo/{qq}") for qq in range(self.num_agent)]
        start_time = time.time()
        i_so_far = start_iteration
        
        # Hàm lưu checkpoint
        def save_checkpoint():
            print("Đang lưu checkpoint...")
            # Thu thập trạng thái hiện tại
            actor_state_dicts = [self.actors[id].state_dict() for id in range(self.num_agent)]
            critic_state_dicts = [self.critics[id].state_dict() for id in range(self.num_agent)]
            optimizer_state_dicts = [self.optimizers[id].state_dict() for id in range(self.num_agent)]
            
            checkpoint = {
                'iteration': i_so_far,
                't_so_far': t_so_far,
                'actor_state_dicts': actor_state_dicts,
                'critic_state_dicts': critic_state_dicts,
                'optimizer_state_dicts': optimizer_state_dicts,
                'loggers': self.loggers.copy() if isinstance(self.loggers, dict) else [logger.copy() for logger in self.loggers],
                'logs': logs
            }
            
            # Lưu vào file tạm trước, sau đó đổi tên để tránh ghi đè không hoàn chỉnh
            temp_checkpoint_path = os.path.join(checkpoint_dir, 'temp_checkpoint.pt')
            torch.save(checkpoint, temp_checkpoint_path)
            if os.path.exists(temp_checkpoint_path):
                os.replace(temp_checkpoint_path, last_checkpoint_path)
                print(f"Đã lưu checkpoint tại vòng lặp {i_so_far}")
            else:
                print(f"Lỗi: Không thể tạo file checkpoint tạm thời")
        
        # Thiết lập bắt tín hiệu ngắt (Ctrl+C, kill, etc.)
        import signal, sys
        
        def signal_handler(sig, frame):
            print('\nQuá trình huấn luyện bị ngắt. Đang lưu checkpoint...')
            save_checkpoint()
            self.file_log_all.close()
            print("Đã đóng tất cả tài nguyên. Thoát chương trình.")
            sys.exit(0)
        
        # Đăng ký bắt tín hiệu
        signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # kill command
        
        # Vòng lặp huấn luyện chính
        while i_so_far <= trained_iterations:
            try:
                # batch_states_all, batch_actions_all, batch_log_probs_all, batch_rewards_all, batch_next_states_all, batch_advantages_all, batch_returns_all, batch_values_all = self.roll_out()
                batch_states_all, batch_actions_all, batch_log_probs_all, batch_rewards_all, batch_next_states_all, batch_advantages_all, batch_returns_all, batch_values_all = self.parallel_roll_out_mp(num_processes=1)
                t_so_far += self.batch_size
                i_so_far += 1
                
                for id in range(self.num_agent):
                    self.loggers[id]['t_so_far'] += self.batch_size
                    self.loggers[id]['i_so_far'] += 1
                    batch_states, batch_actions, batch_log_probs, batch_rewards, batch_next_states, batch_advantages, batch_returns, batch_values = batch_states_all[id], batch_actions_all[id], batch_log_probs_all[id], batch_rewards_all[id], batch_next_states_all[id], batch_advantages_all[id], batch_returns_all[id], batch_values_all[id]
                    b_inds = np.arange(self.batch_size)
                    clipfracs = []
                    for _ in range(self.n_updates_per_iteration):
                        np.random.shuffle(b_inds)
                        for start in range(0, self.batch_size, self.minibatch_size):
                            end = start + self.minibatch_size
                            mb_inds = b_inds[start:end]
                            newlogprob, entropy = self.evaluate(id, batch_states[mb_inds], batch_actions[mb_inds])
                            newvalue = torch.squeeze(self.get_value(id, batch_states[mb_inds]))
                            logratio = newlogprob - batch_log_probs[mb_inds]
                            ratio = logratio.exp()
                            with torch.no_grad():
                                old_approx_kl = (-logratio).mean()
                                approx_kl = ((ratio - 1) - logratio).mean()
                                clipfracs += [((ratio - 1.0).abs() > self.clip).float().mean().item()]
                            mb_advantages = batch_advantages[mb_inds]
                            if self.norm_adv:
                                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                            pg_loss1 = -mb_advantages * ratio
                            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip, 1 + self.clip)
                            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                            newvalue = newvalue.view(-1)
                            if self.clip_vloss:
                                v_loss_unclipped = (newvalue - batch_returns[mb_inds]) ** 2
                                v_clipped = batch_values[mb_inds] + torch.clamp(
                                    newvalue - batch_values[mb_inds],
                                    -self.clip,
                                    self.clip,
                                )
                                v_loss_clipped = (v_clipped - batch_returns[mb_inds]) ** 2
                                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                                v_loss = 0.5 * v_loss_max.mean()
                            else:
                                v_loss = 0.5 * ((newvalue - batch_returns[mb_inds]) ** 2).mean()
                            entropy_loss = entropy.mean()
                            loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef
                            self.optimizers[id].zero_grad()
                            loss.backward()
                            
                            nn.utils.clip_grad_norm_(self.actors[id].parameters(), self.max_grad_norm)
                            nn.utils.clip_grad_norm_(self.critics[id].parameters(), self.max_grad_norm)
                            self.optimizers[id].step()
                        

                            self.loggers[id]['losses'].append(torch.clone(loss).detach().cpu().numpy())
                    y_pred, y_true = batch_values.detach().cpu().numpy(), batch_returns.detach().cpu().numpy()
                    var_y = np.var(y_true)
                    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

                    # Ghi metrics cho TensorBoard
                    writers[id].add_scalar("charts/learning_rate", self.optimizers[id].param_groups[0]["lr"], i_so_far)
                    writers[id].add_scalar("losses/value_loss", v_loss.item(), i_so_far)
                    writers[id].add_scalar("losses/policy_loss", pg_loss.item(), i_so_far)
                    writers[id].add_scalar("losses/entropy", entropy_loss.item(), i_so_far)
                    writers[id].add_scalar("losses/old_approx_kl", old_approx_kl.item(), i_so_far)
                    writers[id].add_scalar("losses/approx_kl", approx_kl.item(), i_so_far)
                    writers[id].add_scalar("losses/clipfrac", np.mean(clipfracs), i_so_far)
                    writers[id].add_scalar("losses/explained_variance", explained_var, i_so_far)
                    print("SPS:", int(i_so_far) / (time.time() - start_time))
                    writers[id].add_scalar("charts/SPS", int(i_so_far) / (time.time() - start_time), i_so_far)
                        
                    # Tóm tắt quá trình huấn luyện
                    logs[id].append(self._log_summary(id))

                    # Lưu mô hình theo lịch trình đã định
                    if self.loggers[id]['i_so_far'] % self.save_freq == 0:
                        folder = os.path.join(save_folder, os.path.join(str(self.loggers[id]['i_so_far']), str(id)))
                        if not os.path.exists(folder):
                            os.makedirs(folder)
                        torch.save(self.actors[id].state_dict(), os.path.join(folder, 'actor.pth'))
                        torch.save(self.critics[id].state_dict(), os.path.join(folder, 'critic.pth'))
                        if self.log_file[id] is not None:
                            shutil.copy(self.log_file[id], folder)
                        with open(os.path.join(folder, 'log.csv'), 'a', newline='') as file:
                            writer_log = csv.writer(file)
                            for row in logs[id]:
                                writer_log.writerow(row)
                        logs[id] = []
                        self.log_file[id] = os.path.join(folder, 'log.csv')
                
                # Lưu checkpoint sau mỗi 10 vòng lặp (có thể điều chỉnh)
                if i_so_far % 10 == 0:
                    save_checkpoint()
                    
            except Exception as e:
                print(f"Lỗi trong quá trình huấn luyện: {e}")
                print("Đang lưu checkpoint trước khi thoát...")
                save_checkpoint()
                raise e
        
        # Lưu checkpoint cuối cùng sau khi hoàn thành huấn luyện
        save_checkpoint()
        self.file_log_all.close()
        
        # Đóng các writers
        for writer in writers:
            writer.close()
        
        print("Quá trình huấn luyện hoàn tất.")   
    def _log_summary(self, id):
        t_so_far = self.loggers[id]['t_so_far']
        i_so_far = self.loggers[id]['i_so_far']
        avg_ep_lens = np.mean(self.loggers[id]['ep_lens'])
        avg_ep_lifetime = np.mean([np.sum(ep_rews) for ep_rews in self.loggers[id]['ep_lifetime']])
        avg_loss = np.mean([losses.mean() for losses in self.loggers[id]['losses']])
        avg_rew = np.mean([rewards.mean() for rewards in self.loggers[id]['rewards']])

        delta_t = self.loggers[id]['delta_t']
        self.loggers[id]['delta_t'] = time.time_ns()
        delta_t = (self.loggers[id]['delta_t'] - delta_t) / 1e9

        # Round decimal places for more aesthetic logging messages
        delta_t = str(round(delta_t, 2))
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_lifetime = str(round(avg_ep_lifetime, 2))
        avg_loss = str(round(avg_loss, 5))
        avg_rew = str(round(avg_rew, 5))

        # Print logging statements
        print(flush=True)
        print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Lifetime: {avg_ep_lifetime}", flush=True)
        print(f"Average Loss: {avg_loss}", flush=True)
        print(f"Average Reward: {avg_rew}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

        # Reset batch-specific logging data
        self.loggers[id]['ep_lens'] = []
        self.loggers[id]['ep_lifetime'] = []  
        self.loggers[id]['losses'] = []
        return [i_so_far, t_so_far, avg_ep_lens, avg_ep_lifetime, avg_loss, avg_rew, delta_t]
        


    def parallel_roll_out(self, num_threads):
        """
        Chạy nhiều luồng với mỗi luồng thực hiện hàm roll_out riêng biệt.
        
        Args:
            num_threads: Số luồng cần chạy song song.
            
        Returns:
            Kết hợp dữ liệu từ tất cả các luồng.
        """
        # Tạo lock để đảm bảo an toàn khi truy cập tài nguyên dùng chung
        network_lock = threading.Lock()
        log_lock = threading.Lock()
        
        # Queue để lưu kết quả từ mỗi luồng
        result_queue = queue.Queue()
        
        # Định nghĩa hàm worker cho mỗi luồng
        def worker(thread_id):
            # Tạo môi trường mới cho luồng này
            thread_env = create_env_for_thread()
            
            # Tạo file log riêng cho mỗi luồng
            thread_file_log = open(f"log_thread_{thread_id}.csv", "w")
            thread_writer_log = csv.writer(thread_file_log)
            
            # Chạy roll_out với tài nguyên riêng cho luồng
            result = roll_out_thread(thread_id, thread_env, thread_writer_log, thread_file_log)
            
            # Đóng file log
            thread_file_log.close()
            
            # Đưa kết quả vào queue
            result_queue.put(result)
        
        def create_env_for_thread():
            """Tạo môi trường mới cho mỗi luồng."""
            return WRSN(scenario_path="physical_env/network/network_scenarios/_50targets_109sensors.yaml",
                   agent_type_path="physical_env/mc/mc_types/default.yaml",
                   num_agent=3, map_size=100, density_map=True)
        
        def roll_out_thread(thread_id, thread_env, thread_writer_log, thread_file_log):
            """Phiên bản roll_out riêng cho mỗi luồng."""
            
            batch_states = [[] for _ in range(self.num_agent)]
            batch_actions = [[] for _ in range(self.num_agent)]
            batch_log_probs = [[] for _ in range(self.num_agent)]
            batch_next_states = [[] for _ in range(self.num_agent)]
            batch_rewards = [[] for _ in range(self.num_agent)]
            batch_returns = [[] for _ in range(self.num_agent)]
            batch_advantages = [[] for _ in range(self.num_agent)]
            batch_values = [[] for _ in range(self.num_agent)]
            
            t = [0 for _ in range(self.num_agent)]
            while True:
                

                states = [[] for _ in range(self.num_agent)]
                actions = [[] for _ in range(self.num_agent)]
                log_probs = [[] for _ in range(self.num_agent)]
                next_states = [[] for _ in range(self.num_agent)]
                rewards = [[] for _ in range(self.num_agent)]
                terminals = [[] for _ in range(self.num_agent)]
                
                request = thread_env.reset()
                cnt = [0 for _ in range(self.num_agent)]
                log_probs_pre = [None for _ in range(self.num_agent)]
                print(f"Thread {thread_id} is rolling out - Number of samples: {t}")
                # sub episode is complete
                while True:
                    
                    # Sử dụng lock khi lấy action để đảm bảo an toàn cho neural network
                    action, log_prob = self.get_action(request["agent_id"], request["state"])
                    
                    log_probs_pre[request["agent_id"]] = log_prob
                    request = thread_env.step(request["agent_id"], action)
                    
                    if request["terminal"]:
                        break
                    if log_probs_pre[request["agent_id"]] is None:
                        continue
                    
                    t[request["agent_id"]] += 1
                    cnt[request["agent_id"]] += 1
                    states[request["agent_id"]].append(request["prev_state"])
                    actions[request["agent_id"]].append(request["input_action"])
                    next_states[request["agent_id"]].append(request["state"])
                    rewards[request["agent_id"]].append(request["reward"])
                    log_probs[request["agent_id"]].append(log_probs_pre[request["agent_id"]])
                    terminals[request["agent_id"]].append(request["terminal"])

                    thread_writer_log.writerow([
                        request["agent_id"],
                        round(request["action"][0], 4),
                        round(request["action"][1], 4),
                        round(request["action"][2], 4)
                    ])
                    thread_file_log.flush()

                for id in range(self.num_agent):
                    if len(states[id]) == 0:
                        continue
                    
                    # Sử dụng lock khi gọi cal_rt_adv để đảm bảo an toàn
                    with network_lock:
                        returns, advantages, values = self.cal_rt_adv(
                            id, 
                            torch.Tensor(np.array(states[id])).to(self.device), 
                            torch.Tensor(np.array(rewards[id])).to(self.device), 
                            torch.Tensor(np.array(next_states[id])).to(self.device), 
                            torch.Tensor(np.array(terminals[id])).to(self.device)
                        )
                    
                    batch_states[id].extend(states[id])
                    batch_actions[id].extend(actions[id])
                    batch_log_probs[id].extend(log_probs[id])
                    batch_next_states[id].extend(next_states[id])
                    batch_rewards[id].extend(rewards[id])
                    batch_advantages[id].extend(advantages)
                    batch_returns[id].extend(returns)
                    batch_values[id].extend(values)

                # Cập nhật logger chung (cần lock để đảm bảo an toàn)
                with log_lock:
                    for id in range(self.num_agent):
                        self.loggers[id]["ep_lens"].append(cnt[id])
                        self.loggers[id]["ep_lifetime"].append(thread_env.env.now)
                        self.loggers[id]["rewards"].append(np.array(rewards[id]))
                
                out = True
                for element in t:
                    if element < self.batch_size/16:
                        out = False
                        break
                if out:
                    break

            # Xử lý dữ liệu batch
            for id in range(self.num_agent):
                if len(batch_rewards[id]) > 0:
                    mean = np.mean(batch_rewards[id])
                    abs_diff = np.abs(batch_rewards[id] - mean)
                    indices = np.argsort(abs_diff)
                    selected_num = int(self.batch_size / 2.0)
                    random_num = self.batch_size - selected_num
                    if len(indices) > selected_num:
                        indices = np.concatenate((indices[-selected_num:], np.random.choice(len(batch_rewards[id]) - selected_num, size=min(random_num, len(batch_rewards[id]) - selected_num), replace=False)))
                    
                    batch_rewards[id] = torch.FloatTensor(np.array(batch_rewards[id])[indices]).to(self.device)
                    batch_states[id] = torch.FloatTensor(np.array(batch_states[id])[indices]).to(self.device)
                    batch_actions[id] = torch.FloatTensor(np.array(batch_actions[id])[indices]).to(self.device)
                    batch_next_states[id] = torch.FloatTensor(np.array(batch_next_states[id])[indices]).to(self.device)
                    batch_log_probs[id] = torch.FloatTensor(np.array(batch_log_probs[id])[indices]).to(self.device)
                    batch_returns[id] = torch.stack([batch_returns[id][_] for _ in indices]).to(self.device)
                    batch_advantages[id] = torch.stack([batch_advantages[id][_] for _ in indices]).to(self.device)
                    batch_values[id] = torch.stack([batch_values[id][_] for _ in indices]).to(self.device)
            
            return batch_states, batch_actions, batch_log_probs, batch_rewards, batch_next_states, batch_advantages, batch_returns, batch_values
        
        # Tạo và khởi động các luồng
        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=worker, args=(i,))
            thread.start()
            threads.append(thread)
        
        # Đợi tất cả các luồng hoàn thành
        for thread in threads:
            thread.join()
        
        # Thu thập kết quả từ tất cả các luồng
        all_results = []
        while not result_queue.empty():
            all_results.append(result_queue.get())
        
        # Kết hợp kết quả từ tất cả các luồng
        combined_batch_states = [[] for _ in range(self.num_agent)]
        combined_batch_actions = [[] for _ in range(self.num_agent)]
        combined_batch_log_probs = [[] for _ in range(self.num_agent)]
        combined_batch_rewards = [[] for _ in range(self.num_agent)]
        combined_batch_next_states = [[] for _ in range(self.num_agent)]
        combined_batch_advantages = [[] for _ in range(self.num_agent)]
        combined_batch_returns = [[] for _ in range(self.num_agent)]
        combined_batch_values = [[] for _ in range(self.num_agent)]
        
        # Kết hợp dữ liệu từ tất cả các luồng
        for result in all_results:
            batch_states, batch_actions, batch_log_probs, batch_rewards, batch_next_states, batch_advantages, batch_returns, batch_values = result
            
            for id in range(self.num_agent):
                if len(batch_rewards[id]) > 0:
                    combined_batch_states[id].append(batch_states[id])
                    combined_batch_actions[id].append(batch_actions[id])
                    combined_batch_log_probs[id].append(batch_log_probs[id])
                    combined_batch_rewards[id].append(batch_rewards[id])
                    combined_batch_next_states[id].append(batch_next_states[id])
                    combined_batch_advantages[id].append(batch_advantages[id])
                    combined_batch_returns[id].append(batch_returns[id])
                    combined_batch_values[id].append(batch_values[id])
        
        # Nếu cần, điều chỉnh kích thước batch cho mỗi tác nhân
        final_batch_states = [[] for _ in range(self.num_agent)]
        final_batch_actions = [[] for _ in range(self.num_agent)]
        final_batch_log_probs = [[] for _ in range(self.num_agent)]
        final_batch_rewards = [[] for _ in range(self.num_agent)]
        final_batch_next_states = [[] for _ in range(self.num_agent)]
        final_batch_advantages = [[] for _ in range(self.num_agent)]
        final_batch_returns = [[] for _ in range(self.num_agent)]
        final_batch_values = [[] for _ in range(self.num_agent)]
        
        for id in range(self.num_agent):
            # Đảm bảo có dữ liệu cho agent này
            if not combined_batch_states[id]:
                continue
                
            # Kết hợp dữ liệu từ tất cả các luồng cho agent này
            if all(isinstance(batch, torch.Tensor) for batch in combined_batch_states[id]):
                final_batch_states[id] = torch.cat(combined_batch_states[id])
                final_batch_actions[id] = torch.cat(combined_batch_actions[id])
                final_batch_log_probs[id] = torch.cat(combined_batch_log_probs[id])
                final_batch_rewards[id] = torch.cat(combined_batch_rewards[id])
                final_batch_next_states[id] = torch.cat(combined_batch_next_states[id])
                final_batch_advantages[id] = torch.cat(combined_batch_advantages[id])
                final_batch_returns[id] = torch.cat(combined_batch_returns[id])
                final_batch_values[id] = torch.cat(combined_batch_values[id])
            
            # Điều chỉnh kích thước batch nếu cần
            if len(final_batch_states[id]) > self.batch_size:
                # Giữ lại batch_size mẫu ngẫu nhiên
                indices = np.random.choice(len(final_batch_states[id]), self.batch_size, replace=False)
                
                final_batch_states[id] = final_batch_states[id][indices]
                final_batch_actions[id] = final_batch_actions[id][indices]
                final_batch_log_probs[id] = final_batch_log_probs[id][indices]
                final_batch_rewards[id] = final_batch_rewards[id][indices]
                final_batch_next_states[id] = final_batch_next_states[id][indices]
                final_batch_advantages[id] = final_batch_advantages[id][indices]
                final_batch_returns[id] = final_batch_returns[id][indices]
                final_batch_values[id] = final_batch_values[id][indices]
        
        return (
            final_batch_states, 
            final_batch_actions, 
            final_batch_log_probs, 
            final_batch_rewards, 
            final_batch_next_states, 
            final_batch_advantages, 
            final_batch_returns, 
            final_batch_values
        )
    
    def parallel_roll_out_mp(self, num_processes):
    
        def create_env_for_process():
            """Tạo môi trường mới cho mỗi luồng."""
            return WRSN(scenario_path="physical_env/network/network_scenarios/_50targets_109sensors.yaml",
                   agent_type_path="physical_env/mc/mc_types/default.yaml",
                   num_agent=3, map_size=100, density_map=True)
        
        def roll_out_process(thread_id, thread_env,thread_writer_log, thread_file_log):
            """Phiên bản roll_out riêng cho mỗi luồng."""
            
            batch_states = [[] for _ in range(self.num_agent)]
            batch_actions = [[] for _ in range(self.num_agent)]
            batch_log_probs = [[] for _ in range(self.num_agent)]
            batch_next_states = [[] for _ in range(self.num_agent)]
            batch_rewards = [[] for _ in range(self.num_agent)]
            batch_returns = [[] for _ in range(self.num_agent)]
            batch_advantages = [[] for _ in range(self.num_agent)]
            batch_values = [[] for _ in range(self.num_agent)]
            
            t = [0 for _ in range(self.num_agent)]
            while True:
                

                states = [[] for _ in range(self.num_agent)]
                actions = [[] for _ in range(self.num_agent)]
                log_probs = [[] for _ in range(self.num_agent)]
                next_states = [[] for _ in range(self.num_agent)]
                rewards = [[] for _ in range(self.num_agent)]
                terminals = [[] for _ in range(self.num_agent)]
                
                request = thread_env.reset()
                cnt = [0 for _ in range(self.num_agent)]
                log_probs_pre = [None for _ in range(self.num_agent)]
                
                # sub episode is complete
                
                while True:
                    print(f"Process  {thread_id} is rolling out - Number of samples: {t}")
                    # Sử dụng lock khi lấy action để đảm bảo an toàn cho neural network

                    action, log_prob = self.get_action(request["agent_id"], request["state"])
                    # action, log_prob = self.get_action(request["agent_id"])
                    
                    print("1")
                    log_probs_pre[request["agent_id"]] = log_prob
                    request = thread_env.step(request["agent_id"], action)
                    
                    if request["terminal"]:
                        break
                    if log_probs_pre[request["agent_id"]] is None:
                        continue
                    
                    t[request["agent_id"]] += 1
                    cnt[request["agent_id"]] += 1
                    states[request["agent_id"]].append(request["prev_state"])
                    actions[request["agent_id"]].append(request["input_action"])
                    next_states[request["agent_id"]].append(request["state"])
                    rewards[request["agent_id"]].append(request["reward"])
                    log_probs[request["agent_id"]].append(log_probs_pre[request["agent_id"]])
                    terminals[request["agent_id"]].append(request["terminal"])

                    thread_writer_log.writerow([
                        request["agent_id"],
                        round(request["action"][0], 4),
                        round(request["action"][1], 4),
                        round(request["action"][2], 4)
                    ])
                    thread_file_log.flush()

                for id in range(self.num_agent):
                    if len(states[id]) == 0:
                        continue
                    
                    # Sử dụng lock khi gọi cal_rt_adv để đảm bảo an toàn
                    returns, advantages, values = self.cal_rt_adv(
                            id, 
                            torch.Tensor(np.array(states[id])), 
                            torch.Tensor(np.array(rewards[id])), 
                            torch.Tensor(np.array(next_states[id])), 
                            torch.Tensor(np.array(terminals[id]))
                        )
                    
                    batch_states[id].extend(states[id])
                    batch_actions[id].extend(actions[id])
                    batch_log_probs[id].extend(log_probs[id])
                    batch_next_states[id].extend(next_states[id])
                    batch_rewards[id].extend(rewards[id])
                    batch_advantages[id].extend(advantages)
                    batch_returns[id].extend(returns)
                    batch_values[id].extend(values)

                # Cập nhật logger chung (cần lock để đảm bảo an toàn)
                for id in range(self.num_agent):
                    self.loggers[id]["ep_lens"].append(cnt[id])
                    self.loggers[id]["ep_lifetime"].append(thread_env.env.now)
                    self.loggers[id]["rewards"].append(np.array(rewards[id]))
                
                out = True
                for element in t:
                    if element < self.batch_size/4:
                        out = False
                        break
                if out:
                    break
            
            print("CHeck ")
            # Xử lý dữ liệu batch
            for id in range(self.num_agent):
                if len(batch_rewards[id]) > 0:
                    mean = np.mean(batch_rewards[id])
                    abs_diff = np.abs(batch_rewards[id] - mean)
                    indices = np.argsort(abs_diff)
                    selected_num = int(self.batch_size / 2.0)
                    random_num = self.batch_size - selected_num
                    if len(indices) > selected_num:
                        indices = np.concatenate((indices[-selected_num:], np.random.choice(len(batch_rewards[id]) - selected_num, size=min(random_num, len(batch_rewards[id]) - selected_num), replace=False)))
                    
                    batch_rewards[id] = torch.FloatTensor(np.array(batch_rewards[id])[indices]).to(self.device)
                    batch_states[id] = torch.FloatTensor(np.array(batch_states[id])[indices]).to(self.device)
                    batch_actions[id] = torch.FloatTensor(np.array(batch_actions[id])[indices]).to(self.device)
                    batch_next_states[id] = torch.FloatTensor(np.array(batch_next_states[id])[indices]).to(self.device)
                    batch_log_probs[id] = torch.FloatTensor(np.array(batch_log_probs[id])[indices]).to(self.device)
                    batch_returns[id] = torch.stack([batch_returns[id][_] for _ in indices]).to(self.device)
                    batch_advantages[id] = torch.stack([batch_advantages[id][_] for _ in indices]).to(self.device)
                    batch_values[id] = torch.stack([batch_values[id][_] for _ in indices]).to(self.device)
            
            return batch_states, batch_actions, batch_log_probs, batch_rewards, batch_next_states, batch_advantages, batch_returns, batch_values
        
            # # Đảm bảo khởi tạo đúng cho PyTorch multiprocessing
        
        # try:
        #     set_start_method('spawn', force=True)
        # except RuntimeError:
        #     pass
        # Queue để lưu kết quả từ các process
        result_queue = Queue()
                    # Tạo mô hình mới riêng cho process này
        proc_actors = [actor.to('cpu') for actor in self.actors]
        proc_critics = [critic.to('cpu') for critic in self.critics]
        # Định nghĩa hàm worker cho process
        def worker(proc_id, result_queue):
            # Tạo môi trường mới cho process
            proc_env = create_env_for_process()
            



            # Tạo mô hình mới riêng cho process này
            proc_actors = [actor for actor in self.actors]
            proc_critics = [critic for critic in self.critics]
            # Tạo file log riêng cho mỗi process
            proc_file_log = open(f"log_process_{proc_id}.csv", "w")
            proc_writer_log = csv.writer(proc_file_log)
            
            # Chạy roll_out với tài nguyên riêng cho process
            result = roll_out_process(
                proc_id, proc_env,  proc_writer_log, proc_file_log
            )
            
            # Đóng file log
            proc_file_log.close()
            
            # Đưa kết quả vào queue
            result_queue.put(result)
        
        # Khởi động các process
        try:
            processes = []
            for i in range(num_processes):
                p = Process(target=worker, args=(i, result_queue))
                p.start()
                processes.append(p)
        

            # Đợi tất cả các process hoàn thành
            for p in processes:
                p.join()
        except KeyboardInterrupt:
            print("Đang dừng các process...")
            for p in processes:
                p.terminate()
            raise KeyboardInterrupt

        all_results = []
        while not result_queue.empty():
            all_results.append(result_queue.get())
        
        # Kết hợp kết quả từ tất cả các luồng
        combined_batch_states = [[] for _ in range(self.num_agent)]
        combined_batch_actions = [[] for _ in range(self.num_agent)]
        combined_batch_log_probs = [[] for _ in range(self.num_agent)]
        combined_batch_rewards = [[] for _ in range(self.num_agent)]
        combined_batch_next_states = [[] for _ in range(self.num_agent)]
        combined_batch_advantages = [[] for _ in range(self.num_agent)]
        combined_batch_returns = [[] for _ in range(self.num_agent)]
        combined_batch_values = [[] for _ in range(self.num_agent)]
        
        # Kết hợp dữ liệu từ tất cả các luồng
        for result in all_results:
            batch_states, batch_actions, batch_log_probs, batch_rewards, batch_next_states, batch_advantages, batch_returns, batch_values = result
            
            for id in range(self.num_agent):
                if len(batch_rewards[id]) > 0:
                    combined_batch_states[id].append(batch_states[id])
                    combined_batch_actions[id].append(batch_actions[id])
                    combined_batch_log_probs[id].append(batch_log_probs[id])
                    combined_batch_rewards[id].append(batch_rewards[id])
                    combined_batch_next_states[id].append(batch_next_states[id])
                    combined_batch_advantages[id].append(batch_advantages[id])
                    combined_batch_returns[id].append(batch_returns[id])
                    combined_batch_values[id].append(batch_values[id])
        
        # Nếu cần, điều chỉnh kích thước batch cho mỗi tác nhân
        final_batch_states = [[] for _ in range(self.num_agent)]
        final_batch_actions = [[] for _ in range(self.num_agent)]
        final_batch_log_probs = [[] for _ in range(self.num_agent)]
        final_batch_rewards = [[] for _ in range(self.num_agent)]
        final_batch_next_states = [[] for _ in range(self.num_agent)]
        final_batch_advantages = [[] for _ in range(self.num_agent)]
        final_batch_returns = [[] for _ in range(self.num_agent)]
        final_batch_values = [[] for _ in range(self.num_agent)]
        
        for id in range(self.num_agent):
            # Đảm bảo có dữ liệu cho agent này
            if not combined_batch_states[id]:
                continue
                
            # Kết hợp dữ liệu từ tất cả các luồng cho agent này
            if all(isinstance(batch, torch.Tensor) for batch in combined_batch_states[id]):
                final_batch_states[id] = torch.cat(combined_batch_states[id])
                final_batch_actions[id] = torch.cat(combined_batch_actions[id])
                final_batch_log_probs[id] = torch.cat(combined_batch_log_probs[id])
                final_batch_rewards[id] = torch.cat(combined_batch_rewards[id])
                final_batch_next_states[id] = torch.cat(combined_batch_next_states[id])
                final_batch_advantages[id] = torch.cat(combined_batch_advantages[id])
                final_batch_returns[id] = torch.cat(combined_batch_returns[id])
                final_batch_values[id] = torch.cat(combined_batch_values[id])
            
            # Điều chỉnh kích thước batch nếu cần
            if len(final_batch_states[id]) > self.batch_size:
                # Giữ lại batch_size mẫu ngẫu nhiên
                indices = np.random.choice(len(final_batch_states[id]), self.batch_size, replace=False)
                
                final_batch_states[id] = final_batch_states[id][indices]
                final_batch_actions[id] = final_batch_actions[id][indices]
                final_batch_log_probs[id] = final_batch_log_probs[id][indices]
                final_batch_rewards[id] = final_batch_rewards[id][indices]
                final_batch_next_states[id] = final_batch_next_states[id][indices]
                final_batch_advantages[id] = final_batch_advantages[id][indices]
                final_batch_returns[id] = final_batch_returns[id][indices]
                final_batch_values[id] = final_batch_values[id][indices]
        
        return (
            final_batch_states, 
            final_batch_actions, 
            final_batch_log_probs, 
            final_batch_rewards, 
            final_batch_next_states, 
            final_batch_advantages, 
            final_batch_returns, 
            final_batch_values
        )

    
    # def parallel_roll_out_mp(self, num_processes):
        # Đảm bảo khởi tạo đúng cho PyTorch multiprocessing
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
        
        # Queue để lưu kết quả từ các process
        result_queue = mp.Queue()
        
        def create_env_for_process():
            """Tạo môi trường mới cho mỗi process."""
            return WRSN(scenario_path="physical_env/network/network_scenarios/_50targets_109sensors.yaml",
                    agent_type_path="physical_env/mc/mc_types/default.yaml",
                    num_agent=3, map_size=100, density_map=True)
        
        # Tạo các tác vụ riêng biệt cho mỗi process
        def worker(proc_id, result_queue):
            # Tạo môi trường mới cho process
            proc_env = create_env_for_process()
            
            # Tạo bản sao của mô hình cho process này - đặt trên CPU
            proc_actors = [copy.deepcopy(actor).to('cpu') for actor in self.actors]
            proc_critics = [copy.deepcopy(critic).to('cpu') for critic in self.critics]
            
            # Tạo file log riêng
            proc_file_log = open(f"log_process_{proc_id}.csv", "w")
            proc_writer_log = csv.writer(proc_file_log)
            
            # Hàm get_action riêng cho process này
            def proc_get_action(agent_id, state_in):
                # Đảm bảo state_in là numpy array
                if not isinstance(state_in, np.ndarray):
                    print(f"Warning: state_in is not a numpy array: {type(state_in)}")
                    state_in = np.array(state_in, dtype=np.float32)
                
                # Chuyển đổi sang tensor trên CPU
                state = torch.from_numpy(state_in).float()
                if state.ndim == 3:
                    state = torch.unsqueeze(state, dim=0)
                
                # Sử dụng mô hình trên CPU
                mean, log_std = proc_actors[agent_id](state)
                std = log_std.exp()
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
                action_log_prob = dist.log_prob(action)
                
                return action.detach().numpy(), action_log_prob.sum().detach().numpy()
            
            # Hàm cal_rt_adv riêng cho process này
            def proc_cal_rt_adv(id, states, rewards, next_states, terminals):
                with torch.no_grad():
                    # Đảm bảo tất cả là tensor trên CPU
                    values = proc_critics[id](states)
                    next_values = proc_critics[id](next_states)
                    
                    if self.gae:
                        advantages = torch.zeros_like(rewards)
                        lastgaelam = 0
                        for t in reversed(range(len(rewards))): 
                            delta = rewards[t] + self.gamma * next_values[t] * int(terminals[t]) - values[t]
                            lastgaelam = delta + self.gamma * self.gae_lambda * int(terminals[t]) * lastgaelam
                            advantages[t] = lastgaelam
                        returns = advantages + values
                    else:
                        returns = torch.zeros_like(rewards)
                        for t in reversed(range(len(rewards))):
                            if t == len(rewards) - 1:
                                next_return = next_values[t]
                            else:
                                next_return = returns[t + 1]
                            returns[t] = rewards[t] + self.gamma * terminals[t] * next_return
                        advantages = returns - values
                    
                    return returns, advantages, values
            
            # Thực hiện roll_out
            batch_states = [[] for _ in range(self.num_agent)]
            batch_actions = [[] for _ in range(self.num_agent)]
            batch_log_probs = [[] for _ in range(self.num_agent)]
            batch_next_states = [[] for _ in range(self.num_agent)]
            batch_rewards = [[] for _ in range(self.num_agent)]
            batch_returns = [[] for _ in range(self.num_agent)]
            batch_advantages = [[] for _ in range(self.num_agent)]
            batch_values = [[] for _ in range(self.num_agent)]
            
            t = [0 for _ in range(self.num_agent)]
            
            while True:
                states = [[] for _ in range(self.num_agent)]
                actions = [[] for _ in range(self.num_agent)]
                log_probs = [[] for _ in range(self.num_agent)]
                next_states = [[] for _ in range(self.num_agent)]
                rewards = [[] for _ in range(self.num_agent)]
                terminals = [[] for _ in range(self.num_agent)]
                
                request = proc_env.reset()
                cnt = [0 for _ in range(self.num_agent)]
                log_probs_pre = [None for _ in range(self.num_agent)]
                
                print(f"Process {proc_id} is rolling out - Samples so far: {t}")
                
                while True:
                    # Sử dụng hàm get_action local
                    action, log_prob = proc_get_action(request["agent_id"], request["state"])
                    
                    log_probs_pre[request["agent_id"]] = log_prob
                    request = proc_env.step(request["agent_id"], action)
                    
                    if request["terminal"]:
                        break
                        
                    if log_probs_pre[request["agent_id"]] is None:
                        continue
                    
                    t[request["agent_id"]] += 1
                    cnt[request["agent_id"]] += 1
                    states[request["agent_id"]].append(request["prev_state"])
                    actions[request["agent_id"]].append(request["input_action"])
                    next_states[request["agent_id"]].append(request["state"])
                    rewards[request["agent_id"]].append(request["reward"])
                    log_probs[request["agent_id"]].append(log_probs_pre[request["agent_id"]])
                    terminals[request["agent_id"]].append(request["terminal"])
                    
                    proc_writer_log.writerow([
                        request["agent_id"],
                        round(request["action"][0], 4),
                        round(request["action"][1], 4),
                        round(request["action"][2], 4)
                    ])
                    proc_file_log.flush()
                
                for id in range(self.num_agent):
                    if len(states[id]) == 0:
                        continue
                    
                    # Chuyển đổi sang tensor trên CPU
                    states_tensor = torch.tensor(np.array(states[id]), dtype=torch.float32)
                    rewards_tensor = torch.tensor(np.array(rewards[id]), dtype=torch.float32)
                    next_states_tensor = torch.tensor(np.array(next_states[id]), dtype=torch.float32)
                    terminals_tensor = torch.tensor(np.array(terminals[id]), dtype=torch.float32)
                    
                    # Sử dụng hàm cal_rt_adv local
                    returns, advantages, values = proc_cal_rt_adv(
                        id, states_tensor, rewards_tensor, next_states_tensor, terminals_tensor
                    )
                    
                    batch_states[id].extend(states[id])
                    batch_actions[id].extend(actions[id])
                    batch_log_probs[id].extend(log_probs[id])
                    batch_next_states[id].extend(next_states[id])
                    batch_rewards[id].extend(rewards[id])
                    batch_advantages[id].extend(advantages.tolist())
                    batch_returns[id].extend(returns.tolist())
                    batch_values[id].extend(values.tolist())
                
                # Kiểm tra có đủ mẫu chưa
                out = True
                for element in t:
                    if element < self.batch_size/num_processes:
                        out = False
                        break
                if out:
                    break
            
            # Xử lý kết quả giống như trong hàm gốc, nhưng giữ tất cả trên CPU
            for id in range(self.num_agent):
                if len(batch_rewards[id]) > 0:
                    # Xử lý lựa chọn mẫu
                    mean = np.mean(batch_rewards[id])
                    abs_diff = np.abs(np.array(batch_rewards[id]) - mean)
                    indices = np.argsort(abs_diff)
                    selected_num = int(self.batch_size / (2.0 * num_processes))
                    random_num = self.batch_size // num_processes - selected_num
                    
                    if len(indices) > selected_num:
                        indices = np.concatenate((
                            indices[-selected_num:], 
                            np.random.choice(len(batch_rewards[id]) - selected_num, 
                                            size=min(random_num, len(batch_rewards[id]) - selected_num), 
                                            replace=False)
                        ))
                    
                    # Chuyển sang tensor cho kết quả final
                    batch_rewards[id] = torch.FloatTensor(np.array(batch_rewards[id])[indices])
                    batch_states[id] = torch.FloatTensor(np.array(batch_states[id])[indices])
                    batch_actions[id] = torch.FloatTensor(np.array(batch_actions[id])[indices])
                    batch_next_states[id] = torch.FloatTensor(np.array(batch_next_states[id])[indices])
                    batch_log_probs[id] = torch.FloatTensor(np.array(batch_log_probs[id])[indices])
                    
                    # Chuyển các tensor list thành tensor
                    batch_returns[id] = torch.FloatTensor([batch_returns[id][_] for _ in indices])
                    batch_advantages[id] = torch.FloatTensor([batch_advantages[id][_] for _ in indices])
                    batch_values[id] = torch.FloatTensor([batch_values[id][_] for _ in indices])
            
            # Trả về kết quả để kết hợp
            result_queue.put((batch_states, batch_actions, batch_log_probs, 
                            batch_rewards, batch_next_states, batch_advantages, 
                            batch_returns, batch_values))

            # Đóng file log
            proc_file_log.close()
            
            # Đưa kết quả vào queue
            result_queue.put(result)
        
        # Khởi động các process
        try:
            processes = []
            for i in range(num_processes):
                p = Process(target=worker, args=(i, result_queue))
                p.start()
                processes.append(p)
        

            # Đợi tất cả các process hoàn thành
            for p in processes:
                p.join()
        except KeyboardInterrupt:
            print("Đang dừng các process...")
            for p in processes:
                p.terminate()
            raise KeyboardInterrupt

        all_results = []
        while not result_queue.empty():
            all_results.append(result_queue.get())
        
        # Kết hợp kết quả từ tất cả các luồng
        combined_batch_states = [[] for _ in range(self.num_agent)]
        combined_batch_actions = [[] for _ in range(self.num_agent)]
        combined_batch_log_probs = [[] for _ in range(self.num_agent)]
        combined_batch_rewards = [[] for _ in range(self.num_agent)]
        combined_batch_next_states = [[] for _ in range(self.num_agent)]
        combined_batch_advantages = [[] for _ in range(self.num_agent)]
        combined_batch_returns = [[] for _ in range(self.num_agent)]
        combined_batch_values = [[] for _ in range(self.num_agent)]
        
        # Kết hợp dữ liệu từ tất cả các luồng
        for result in all_results:
            batch_states, batch_actions, batch_log_probs, batch_rewards, batch_next_states, batch_advantages, batch_returns, batch_values = result
            
            for id in range(self.num_agent):
                if len(batch_rewards[id]) > 0:
                    combined_batch_states[id].append(batch_states[id])
                    combined_batch_actions[id].append(batch_actions[id])
                    combined_batch_log_probs[id].append(batch_log_probs[id])
                    combined_batch_rewards[id].append(batch_rewards[id])
                    combined_batch_next_states[id].append(batch_next_states[id])
                    combined_batch_advantages[id].append(batch_advantages[id])
                    combined_batch_returns[id].append(batch_returns[id])
                    combined_batch_values[id].append(batch_values[id])
        
        # Nếu cần, điều chỉnh kích thước batch cho mỗi tác nhân
        final_batch_states = [[] for _ in range(self.num_agent)]
        final_batch_actions = [[] for _ in range(self.num_agent)]
        final_batch_log_probs = [[] for _ in range(self.num_agent)]
        final_batch_rewards = [[] for _ in range(self.num_agent)]
        final_batch_next_states = [[] for _ in range(self.num_agent)]
        final_batch_advantages = [[] for _ in range(self.num_agent)]
        final_batch_returns = [[] for _ in range(self.num_agent)]
        final_batch_values = [[] for _ in range(self.num_agent)]
        
        for id in range(self.num_agent):
            # Đảm bảo có dữ liệu cho agent này
            if not combined_batch_states[id]:
                continue
                
            # Kết hợp dữ liệu từ tất cả các luồng cho agent này
            if all(isinstance(batch, torch.Tensor) for batch in combined_batch_states[id]):
                final_batch_states[id] = torch.cat(combined_batch_states[id])
                final_batch_actions[id] = torch.cat(combined_batch_actions[id])
                final_batch_log_probs[id] = torch.cat(combined_batch_log_probs[id])
                final_batch_rewards[id] = torch.cat(combined_batch_rewards[id])
                final_batch_next_states[id] = torch.cat(combined_batch_next_states[id])
                final_batch_advantages[id] = torch.cat(combined_batch_advantages[id])
                final_batch_returns[id] = torch.cat(combined_batch_returns[id])
                final_batch_values[id] = torch.cat(combined_batch_values[id])
            
            # Điều chỉnh kích thước batch nếu cần
            if len(final_batch_states[id]) > self.batch_size:
                # Giữ lại batch_size mẫu ngẫu nhiên
                indices = np.random.choice(len(final_batch_states[id]), self.batch_size, replace=False)
                
                final_batch_states[id] = final_batch_states[id][indices]
                final_batch_actions[id] = final_batch_actions[id][indices]
                final_batch_log_probs[id] = final_batch_log_probs[id][indices]
                final_batch_rewards[id] = final_batch_rewards[id][indices]
                final_batch_next_states[id] = final_batch_next_states[id][indices]
                final_batch_advantages[id] = final_batch_advantages[id][indices]
                final_batch_returns[id] = final_batch_returns[id][indices]
                final_batch_values[id] = final_batch_values[id][indices]
        
        return (
            final_batch_states, 
            final_batch_actions, 
            final_batch_log_probs, 
            final_batch_rewards, 
            final_batch_next_states, 
            final_batch_advantages, 
            final_batch_returns, 
            final_batch_values
        )