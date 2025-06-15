import os
import numpy as np
import torch
import torch.nn as nn
import time
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from controller.ippo.actor.UnetActor import UNet
from controller.ippo.critic.CNNCritic import CNNCritic

import shutil
import csv
from rl_env.WRSN import WRSN
import time
import threading
import queue
import copy
from torch.multiprocessing import Process
import os
import csv
from torch.multiprocessing import Queue, Process, set_start_method, Manager
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system') 
from torch.multiprocessing import Manager
def create_environment():
        return WRSN(scenario_path="physical_env/network/network_scenarios/50_target_train.yaml",
                   agent_type_path="physical_env/mc/mc_types/default.yaml",
                   num_agent=3, map_size=100, density_map=True)

class IPPO:
    def __init__(self, args, env, device, model_path=None):
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

    
    def get_global_param(self):
        return self

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
        state = torch.FloatTensor(state_in).to(self.device)
        if state.ndim == 3:
            state = torch.unsqueeze(state, dim=0)
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
    
    
    def train(self, trained_iterations, save_folder):
        # Kiểm tra và tạo thư mục lưu trữ nếu chưa tồn tại
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

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
        
        # Hàm lưu checkpoint CHỈ sau khi hoàn thành một iteration
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
        
        # Thiết lập bắt tín hiệu ngắt (Ctrl+C, kill, etc.) - KHÔNG lưu checkpoint khi bị ngắt
        import signal, sys
        
        def signal_handler(sig, frame):
            print('\nQuá trình huấn luyện bị ngắt. Thoát chương trình mà không lưu checkpoint hiện tại.')
            self.file_log_all.close()
            print("Đã đóng tất cả tài nguyên. Thoát chương trình.")
            sys.exit(0)
        
        # Đăng ký bắt tín hiệu
        signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # kill command
        
        # Cờ kiểm tra iteration đã hoàn thành hay chưa
        iteration_completed = False
        
        # Vòng lặp huấn luyện chính
        while i_so_far <= trained_iterations:
            try:
                # Đánh dấu iteration chưa hoàn thành
                iteration_completed = False
                
                # batch_states_all, batch_actions_all, batch_log_probs_all, batch_rewards_all, batch_next_states_all, batch_advantages_all, batch_returns_all, batch_values_all = self.roll_out()
                batch_states_all, batch_actions_all, batch_log_probs_all, batch_rewards_all, batch_next_states_all, batch_advantages_all, batch_returns_all, batch_values_all = self.parallel_roll_out_mp(num_processes=8)
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
                            # Thêm kiểm tra và chuyển đổi kiểu dữ liệu
                            if not isinstance(batch_states, torch.Tensor):
                                batch_states = torch.tensor(batch_states, device=self.device)
                            if not isinstance(batch_actions, torch.Tensor):
                                batch_actions = torch.tensor(batch_actions, device=self.device)
                            if not isinstance(mb_inds, np.ndarray):
                                mb_inds = np.array(mb_inds, dtype=np.int64)
                            
                            # Chuyển đổi mb_inds thành tensor nếu batch_states là tensor
                            if isinstance(batch_states, torch.Tensor):
                                mb_inds = torch.tensor(mb_inds, device=batch_states.device, dtype=torch.long)
                            
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
                
                # Đánh dấu iteration đã hoàn thành
                iteration_completed = True
                
                if iteration_completed:
                    # Lưu checkpoint SAU khi hoàn thành một iteration
                    save_checkpoint()
                    
            except Exception as e:
                print(f"Lỗi trong quá trình huấn luyện: {e}")
                # Nếu iteration đã hoàn thành thì đã lưu checkpoint rồi
                # Nếu iteration chưa hoàn thành thì không lưu checkpoint
                print("Thoát chương trình mà không lưu checkpoint mới.")
                raise e
        
        # Đóng file log
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

        # write logging statements to file on save


        # Reset batch-specific logging data
        self.loggers[id]['ep_lens'] = []
        self.loggers[id]['ep_lifetime'] = []  
        self.loggers[id]['losses'] = []
        return [i_so_far, t_so_far, avg_ep_lens, avg_ep_lifetime, avg_loss, avg_rew, delta_t]



    def parallel_roll_out_mp(self, num_processes):
        """
        Chạy nhiều process song song, mỗi process thu thập batch_size/num_processes mẫu.
        
        Args:
            num_processes: Số process chạy song song.
            
        Returns:
            Kết hợp dữ liệu từ tất cả các process.
        """
        # Khởi tạo multiprocessing
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
        
        # Queue để lưu kết quả
        result_queue = mp.Queue()
        manager = Manager()
        result_list = manager.list()
        
        # Chuẩn bị dữ liệu cho worker
        actor_states = [actor.state_dict() for actor in self.actors]
        critic_states = [critic.state_dict() for critic in self.critics]
        
        # Khởi động các process
        processes = []
        for i in range(num_processes):
            p = Process(
                target=standalone_worker, 
                args=(
                    i, result_list, actor_states, critic_states, 
                    self.num_agent, self.batch_size, num_processes,
                    self.gamma, self.gae_lambda, self.gae, self.device
                )
            )
            p.daemon = True
            p.start()
            processes.append(p)
        
        # Đợi tất cả các process hoàn thành
        for p in processes:
            p.join()
            # Thu thập kết quả
            all_results = list(result_list)
            
            
            # Kết hợp kết quả từ tất cả các process
            combined_batch_states = [[] for _ in range(self.num_agent)]
            combined_batch_actions = [[] for _ in range(self.num_agent)]
            combined_batch_log_probs = [[] for _ in range(self.num_agent)]
            combined_batch_rewards = [[] for _ in range(self.num_agent)]
            combined_batch_next_states = [[] for _ in range(self.num_agent)]
            combined_batch_advantages = [[] for _ in range(self.num_agent)]
            combined_batch_returns = [[] for _ in range(self.num_agent)]
            combined_batch_values = [[] for _ in range(self.num_agent)]
            
            for result in all_results:
                batch_states, batch_actions, batch_log_probs, batch_rewards, batch_next_states, batch_advantages, batch_returns, batch_values,episode_stats = result
                
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
                for id in range(self.num_agent):
                    self.loggers[id]['ep_lens'].extend(episode_stats['ep_lens'][id])
                    self.loggers[id]['ep_lifetime'].extend(episode_stats['ep_lifetime'][id])
                    self.loggers[id]['rewards'].extend(episode_stats['ep_rewards'][id])

        # Kết hợp và chuyển sang device chính
        final_batch_states = [[] for _ in range(self.num_agent)]
        final_batch_actions = [[] for _ in range(self.num_agent)]
        final_batch_log_probs = [[] for _ in range(self.num_agent)]
        final_batch_rewards = [[] for _ in range(self.num_agent)]
        final_batch_next_states = [[] for _ in range(self.num_agent)]
        final_batch_advantages = [[] for _ in range(self.num_agent)]
        final_batch_returns = [[] for _ in range(self.num_agent)]
        final_batch_values = [[] for _ in range(self.num_agent)]
        
        for id in range(self.num_agent):
            if not combined_batch_states[id]:
                continue
                
            if all(isinstance(batch, torch.Tensor) for batch in combined_batch_states[id]):
                final_batch_states[id] = torch.cat(combined_batch_states[id]).to(self.device)
                final_batch_actions[id] = torch.cat(combined_batch_actions[id]).to(self.device)
                final_batch_log_probs[id] = torch.cat(combined_batch_log_probs[id]).to(self.device)
                final_batch_rewards[id] = torch.cat(combined_batch_rewards[id]).to(self.device)
                final_batch_next_states[id] = torch.cat(combined_batch_next_states[id]).to(self.device)
                final_batch_advantages[id] = torch.cat(combined_batch_advantages[id]).to(self.device)
                final_batch_returns[id] = torch.cat(combined_batch_returns[id]).to(self.device)
                final_batch_values[id] = torch.cat(combined_batch_values[id]).to(self.device)
            
            # Điều chỉnh kích thước batch nếu cần
            if len(final_batch_states[id]) > self.batch_size and len(final_batch_advantages[id]) > self.batch_size:
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

def standalone_worker(proc_id, result_queue, actor_states, critic_states, num_agent, batch_size, 
                     num_processes, gamma, gae_lambda, gae, device):

    seed = np.random.randint(0, 10000)

    process_random = np.random.default_rng(seed)
    proc_file_log = open(f"log_process_{proc_id}.csv", "w", newline='')
    proc_writer_log = csv.writer(proc_file_log)
    
    # Tạo môi trường mới cho process
    from rl_env.WRSN import WRSN
    proc_env = WRSN(scenario_path="physical_env/network/network_scenarios/50_target_train.yaml",
                   agent_type_path="physical_env/mc/mc_types/default.yaml",
                   num_agent=num_agent, map_size=100, density_map=True)
    
    # Tạo bản sao của mô hình cho process này
    from controller.ippo.actor.UnetActor import UNet
    from controller.ippo.critic.CNNCritic import CNNCritic
    
    proc_actors = []
    proc_critics = []
    
    for i in range(num_agent):
        actor = UNet()
        actor.load_state_dict(actor_states[i])
        actor.to('cpu')
        proc_actors.append(actor)
        
        critic = CNNCritic()
        critic.load_state_dict(critic_states[i])
        critic.to('cpu')
        proc_critics.append(critic)
    
    # Hàm get_value riêng cho process
    def proc_get_value(agent_id, state):
        value = proc_critics[agent_id](state)
        return value.sum(1)
    
    # Hàm cal_rt_adv riêng cho process
    def proc_cal_rt_adv(id, states, rewards, next_states, terminals):
        with torch.no_grad():
            values = proc_get_value(id, states)
            next_values = proc_get_value(id, next_states)
            
            if gae:
                advantages = torch.zeros_like(rewards)
                lastgaelam = 0
                for t in reversed(range(len(rewards))): 
                    delta = rewards[t] + gamma * next_values[t] * int(terminals[t]) - values[t]
                    lastgaelam = delta + gamma * gae_lambda * int(terminals[t]) * lastgaelam
                    advantages[t] = lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards)
                for t in reversed(range(len(rewards))):
                    if t == len(rewards) - 1:  # Sửa lại điều kiện này
                        next_return = next_values[t]
                    else:
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + gamma * terminals[t] * next_return
                advantages = returns - values
            
            # Đảm bảo trả về các tensors 1D
            returns = returns.view(-1)
            advantages = advantages.view(-1)
            values = values.view(-1)
            
            return returns, advantages, values
    
    # Hàm get_action cho process
    def proc_get_action(agent_id, state_in):
        if not isinstance(state_in, np.ndarray):
            state_in = np.array(state_in, dtype=np.float32)
        
        state = torch.from_numpy(state_in).float()
        if state.ndim == 3:
            state = torch.unsqueeze(state, dim=0)
        
        mean, log_std = proc_actors[agent_id](state)
        std = log_std.exp()
        dist = Normal(mean, std)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        
        return action.detach().numpy(), action_log_prob.sum().detach().numpy()
    
    # Khởi tạo các batch
    batch_states = [[] for _ in range(num_agent)]
    batch_actions = [[] for _ in range(num_agent)]
    batch_log_probs = [[] for _ in range(num_agent)]
    batch_next_states = [[] for _ in range(num_agent)]
    batch_rewards = [[] for _ in range(num_agent)]
    batch_returns = [[] for _ in range(num_agent)]
    batch_advantages = [[] for _ in range(num_agent)]
    batch_values = [[] for _ in range(num_agent)]
    
    # Mỗi process thu thập batch_size / num_processes mẫu
    process_batch_size = batch_size // num_processes
    
    # Đếm số mẫu đã thu thập
    t = [0 for _ in range(num_agent)]
    
    try:
        # Thu thập mẫu
        while True:
            states = [[] for _ in range(num_agent)]
            actions = [[] for _ in range(num_agent)]
            log_probs = [[] for _ in range(num_agent)]
            next_states = [[] for _ in range(num_agent)]
            rewards = [[] for _ in range(num_agent)]
            terminals = [[] for _ in range(num_agent)]
            
            request = proc_env.reset()
            cnt = [0 for _ in range(num_agent)]
            log_probs_pre = [None for _ in range(num_agent)]
            
            print(f"Process {proc_id} is rolling out - Samples so far: {t}")
            
            # Một episode
            while True:
                # Lấy action
                action, log_prob = proc_get_action(request["agent_id"], request["state"])
                
                log_probs_pre[request["agent_id"]] = log_prob
                request = proc_env.step(request["agent_id"], action)
                
                if request["terminal"]:
                    break
                    
                if log_probs_pre[request["agent_id"]] is None:
                    continue
                
                # Thu thập dữ liệu
                t[request["agent_id"]] += 1
                cnt[request["agent_id"]] += 1
                states[request["agent_id"]].append(request["prev_state"])
                actions[request["agent_id"]].append(request["input_action"])
                next_states[request["agent_id"]].append(request["state"])
                rewards[request["agent_id"]].append(request["reward"])
                log_probs[request["agent_id"]].append(log_probs_pre[request["agent_id"]])
                terminals[request["agent_id"]].append(request["terminal"])
                
                # Ghi log
                proc_writer_log.writerow([
                    request["agent_id"],
                    round(request["action"][0], 4),
                    round(request["action"][1], 4),
                    round(request["action"][2], 4)
                ])
                proc_file_log.flush()
            
            # Xử lý sau mỗi episode
            for id in range(num_agent):
                if len(states[id]) == 0:
                    continue
                
                # Chuyển sang tensor
                states_tensor = torch.tensor(np.array(states[id]), dtype=torch.float32)
                rewards_tensor = torch.tensor(np.array(rewards[id]), dtype=torch.float32)
                next_states_tensor = torch.tensor(np.array(next_states[id]), dtype=torch.float32)
                terminals_tensor = torch.tensor(np.array(terminals[id]), dtype=torch.float32)
                
                # Tính returns và advantages bằng hàm proc_cal_rt_adv
                returns, advantages, values = proc_cal_rt_adv(
                    id, states_tensor, rewards_tensor, next_states_tensor, terminals_tensor
                )
                
                # Lưu trữ - đảm bảo là numpy arrays hoặc số, tránh dùng tensor cho list
                batch_states[id].extend(states[id])
                batch_actions[id].extend(actions[id])
                batch_log_probs[id].extend(log_probs[id])
                batch_next_states[id].extend(next_states[id])
                batch_rewards[id].extend(rewards[id])
                batch_advantages[id].extend(advantages.cpu().numpy())
                batch_returns[id].extend(returns.cpu().numpy())
                batch_values[id].extend(values.cpu().numpy())
            
            # Kiểm tra đủ mẫu chưa
            out = True
            for element in t:
                if element < process_batch_size:
                    out = False
                    break
            if out:
                break
        
        # Xử lý batch
        for id in range(num_agent):
            if len(batch_rewards[id]) > 0:
                # Chuyển tất cả sang numpy arrays
                batch_rewards_np = np.array(batch_rewards[id])
                batch_states_np = np.array(batch_states[id])
                batch_actions_np = np.array(batch_actions[id])
                batch_next_states_np = np.array(batch_next_states[id])
                batch_log_probs_np = np.array(batch_log_probs[id])
                batch_returns_np = np.array(batch_returns[id])
                batch_advantages_np = np.array(batch_advantages[id])
                batch_values_np = np.array(batch_values[id])
                
                # Lựa chọn mẫu
                mean = np.mean(batch_rewards_np)
                abs_diff = np.abs(batch_rewards_np - mean)
                indices = np.argsort(abs_diff)
                selected_num = int(process_batch_size / 2.0)
                random_num = process_batch_size - selected_num
                
                if len(indices) > selected_num:
                    selected_indices = indices[-selected_num:]
                    if len(indices) - selected_num > 0:
                        random_indices = process_random.choice(
                            len(batch_rewards_np) - selected_num, 
                            size=min(random_num, len(batch_rewards_np) - selected_num), 
                            replace=False
                        )
                        # Đảm bảo random_indices trỏ vào vị trí đúng
                        mask = np.ones(len(batch_rewards_np), dtype=bool)
                        mask[selected_indices] = False
                        remaining_indices = np.arange(len(batch_rewards_np))[mask]
                        random_chosen = remaining_indices[random_indices]
                        combined_indices = np.concatenate((selected_indices, random_chosen))
                    else:
                        combined_indices = selected_indices
                else:
                    combined_indices = indices
                
                # Chuyển sang tensor
                batch_rewards[id] = torch.FloatTensor(batch_rewards_np[combined_indices])
                batch_states[id] = torch.FloatTensor(batch_states_np[combined_indices])
                batch_actions[id] = torch.FloatTensor(batch_actions_np[combined_indices])
                batch_next_states[id] = torch.FloatTensor(batch_next_states_np[combined_indices])
                batch_log_probs[id] = torch.FloatTensor(batch_log_probs_np[combined_indices])
                batch_returns[id] = torch.FloatTensor(batch_returns_np[combined_indices])
                batch_advantages[id] = torch.FloatTensor(batch_advantages_np[combined_indices])
                batch_values[id] = torch.FloatTensor(batch_values_np[combined_indices])
        # Lưu thông tin thống kê episode
        episode_stats = {
            'ep_lens': [[] for _ in range(num_agent)],
            'ep_lifetime': [[] for _ in range(num_agent)],
            'ep_rewards': [[] for _ in range(num_agent)]
        }

        # Sau mỗi episode, cập nhật thống kê
        for id in range(num_agent):
            episode_stats['ep_lens'][id].append(cnt[id])
            episode_stats['ep_lifetime'][id].append(proc_env.env.now)
            episode_stats['ep_rewards'][id].append(np.array(rewards[id]))
            
        # Trả về kết quả
        result_queue.append((batch_states, batch_actions, batch_log_probs, 
                         batch_rewards, batch_next_states, batch_advantages, 
                         batch_returns, batch_values, episode_stats))
    
    except Exception as e:
        print(f"Lỗi trong process {proc_id}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Đóng file log
        proc_file_log.close()
