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
from rl_env.WRSN import WRSN
import copy
import shutil
import csv
import concurrent.futures
import numpy as np
import copy
from torch.distributions import Normal

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
        # Tăng kích thước batch và xử lý ít lần hơn
        total_batch = batch_states.size(0)
        sub_batch_size = min(256, total_batch)  # Kích thước batch tối ưu cho GPU
        
        action_log_probs = []
        entropies = []
        
        for start_idx in range(0, total_batch, sub_batch_size):
            end_idx = min(start_idx + sub_batch_size, total_batch)
            sub_states = batch_states[start_idx:end_idx]
            sub_actions = batch_actions[start_idx:end_idx]
            
            mean, log_std = self.actors[agent_id](sub_states)
            std = torch.exp(log_std)
            dist = Normal(mean, std)
            sub_log_probs = dist.log_prob(sub_actions)
            
            action_log_probs.append(sub_log_probs.sum((1, 2)))
            entropies.append(dist.entropy().sum((1, 2)))
        
        return torch.cat(action_log_probs), torch.cat(entropies)

    def get_value(self, agent_id, state):
        value = self.critics[agent_id](state)
        return value.sum(1)
    
    # Thu thập dữ liệu cho đến khi đạt được batch_size
    def collection_data(self):
        # create a local copy of the environment
        env = WRSN(scenario_path="physical_env/network/network_scenarios/hanoi1000n50.yaml"
               ,agent_type_path="physical_env/mc/mc_types/default.yaml"
               ,num_agent=3, map_size=100, density_map=True)
        # reset the environment
        request = env.reset()

    # Tạo bộ nhớ cục bộ cho episode này
        states_local = [[] for _ in range(self.num_agent)]
        actions_local = [[] for _ in range(self.num_agent)]
        log_probs_local = [[] for _ in range(self.num_agent)]
        next_states_local = [[] for _ in range(self.num_agent)]
        rewards_local = [[] for _ in range(self.num_agent)]
        terminals_local = [[] for _ in range(self.num_agent)]
        
        cnt = [0 for _ in range(self.num_agent)]
        log_probs_pre = [None for _ in range(self.num_agent)]
        
        while True:
            # Lấy hành động từ mô hình actor
            agent_id = request["agent_id"]
            state = torch.FloatTensor(request["state"]).to(self.device)
            if state.ndim == 3:
                state = torch.unsqueeze(state, dim=0)
            
            with torch.no_grad():
                mean, log_std = self.actors[agent_id](state)
                std = log_std.exp()
                dist = Normal(mean, std)
                action = dist.sample()
                action_log_prob = dist.log_prob(action)
            
            # Chuyển tensor sang numpy để tương tác với môi trường
            action_np = action.detach().cpu().numpy()
            log_prob_np = action_log_prob.sum().detach().cpu().numpy()
            
            log_probs_pre[agent_id] = log_prob_np
            
            # Thực hiện bước trong môi trường
            request = self.env.step(agent_id, action_np)
            
            # Kiểm tra kết thúc episode
            if request["terminal"]:
                break
                
            if log_probs_pre[request["agent_id"]] is None:
                continue
                
            cnt[request["agent_id"]] += 1
            
            # Thu thập dữ liệu
            states_local[request["agent_id"]].append(request["prev_state"])
            actions_local[request["agent_id"]].append(request["input_action"])
            next_states_local[request["agent_id"]].append(request["state"])
            rewards_local[request["agent_id"]].append(request["reward"])
            log_probs_local[request["agent_id"]].append(log_probs_pre[request["agent_id"]])
            terminals_local[request["agent_id"]].append(request["terminal"])
        
        return states_local, actions_local, log_probs_local, next_states_local, rewards_local, terminals_local, cnt


    def roll_out(self):
        # Không còn sao chép môi trường - thay vào đó sử dụng một môi trường duy nhất
        batch_states = [[] for _ in range(self.num_agent)]
        batch_actions = [[] for _ in range(self.num_agent)]
        batch_log_probs = [[] for _ in range(self.num_agent)]
        batch_next_states = [[] for _ in range(self.num_agent)]
        batch_rewards = [[] for _ in range(self.num_agent)]
        batch_returns = [[] for _ in range(self.num_agent)]
        batch_advantages = [[] for _ in range(self.num_agent)]
        batch_values = [[] for _ in range(self.num_agent)]
        
        t = [0 for _ in range(self.num_agent)]
        
        # Thu thập dữ liệu cho đến khi đạt được batch_size
        while True:
            # print sample count
            print("Sample count: ", t)

            # 4 threads paralle to collect data with each self.collect_data is a thread
            
            # Sử dụng ThreadPoolExecutor để chạy 4 luồng song song
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                # Tạo và submit các thread
                futures = [executor.submit(self.collection_data_single_thread, thread_id) 
                        for thread_id in range(self.num_threads)]
                
                # Đợi tất cả các thread hoàn thành và thu thập kết quả
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            

            

            
            # Thu thập dữ liệu từ môi trường
            states_local, actions_local, log_probs_local, next_states_local, rewards_local, terminals_local, cnt = self.collection_data()
            print("Sample count: ", cnt)
            
            # Xử lý dữ liệu thu thập được từ episode này
            for id in range(self.num_agent):
                if len(states_local[id]) == 0:
                    continue
                    
                # Chuyển dữ liệu sang tensor và tính advantages, returns
                states_tensor = torch.FloatTensor(np.array(states_local[id])).to(self.device)
                next_states_tensor = torch.FloatTensor(np.array(next_states_local[id])).to(self.device)
                rewards_tensor = torch.FloatTensor(np.array(rewards_local[id])).to(self.device)
                terminals_tensor = torch.FloatTensor(np.array(terminals_local[id])).to(self.device)
                
                # Tính returns, advantages và values
                returns, advantages, values = self.cal_rt_adv(
                    id, 
                    states_tensor, 
                    rewards_tensor, 
                    next_states_tensor, 
                    terminals_tensor
                )
                
                # Thêm dữ liệu vào batch
                batch_states[id].extend(states_local[id])
                batch_actions[id].extend(actions_local[id])
                batch_log_probs[id].extend(log_probs_local[id])
                batch_next_states[id].extend(next_states_local[id])
                batch_rewards[id].extend(rewards_local[id])
                batch_returns[id].extend(returns)
                batch_advantages[id].extend(advantages)
                batch_values[id].extend(values)
                
                t[id] += len(states_local[id])
                
                # Cập nhật logger
                self.loggers[id]["ep_lens"].append(cnt)
                self.loggers[id]["ep_lifetime"].append(self.env.env.now)
                if len(rewards_local[id]) > 0:
                    self.loggers[id]["rewards"].append(np.array(rewards_local[id]))
            
            # Kiểm tra đủ mẫu chưa
            enough_samples = True
            for id in range(self.num_agent):
                if t[id] < self.batch_size:
                    enough_samples = False
                    break
                    
            if enough_samples:
                break
        
        
        # Xử lý các batch đã thu thập - chọn lọc dữ liệu
        for id in range(self.num_agent):
            if len(batch_rewards[id]) == 0:
                continue
                
            # Tính trung bình rewards và chọn lọc dữ liệu
            mean = np.mean(batch_rewards[id])
            abs_diff = np.abs(batch_rewards[id] - mean)
            indices = np.argsort(abs_diff)
            
            # Chọn một nửa điểm gần với giá trị trung bình và một nửa ngẫu nhiên
            selected_num = int(self.batch_size/2)
            random_num = self.batch_size - selected_num
            
            # Nếu không đủ dữ liệu, lấy hết
            if len(batch_rewards[id]) <= self.batch_size:
                indices = np.arange(len(batch_rewards[id]))
            else:
                # Kết hợp dữ liệu từ top-half gần với trung bình và các mẫu ngẫu nhiên
                indices = np.concatenate((
                    indices[:selected_num],
                    np.random.choice(
                        np.setdiff1d(np.arange(len(batch_rewards[id])), indices[:selected_num]),
                        size=min(random_num, len(batch_rewards[id]) - selected_num),
                        replace=False
                    )
                ))
            
            # Chuyển dữ liệu sang tensor và chọn các mẫu đã lọc
            batch_rewards[id] = torch.FloatTensor(np.array(batch_rewards[id])[indices]).to(self.device)
            batch_states[id] = torch.FloatTensor(np.array(batch_states[id])[indices]).to(self.device)
            batch_actions[id] = torch.FloatTensor(np.array(batch_actions[id])[indices]).to(self.device)
            batch_next_states[id] = torch.FloatTensor(np.array(batch_next_states[id])[indices]).to(self.device)
            batch_log_probs[id] = torch.FloatTensor(np.array(batch_log_probs[id])[indices]).to(self.device)
            
            # Xử lý returns, advantages, values
            batch_returns[id] = torch.stack([batch_returns[id][i] for i in indices]).to(self.device)
            batch_advantages[id] = torch.stack([batch_advantages[id][i] for i in indices]).to(self.device)
            batch_values[id] = torch.stack([batch_values[id][i] for i in indices]).to(self.device)
        
        return batch_states, batch_actions, batch_log_probs, batch_rewards, batch_next_states, batch_advantages, batch_returns, batch_values
    
    
    
    def train(self, trained_iterations, save_folder, resume=True):
        # Kiểm tra và tạo thư mục lưu nếu chưa tồn tại
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
        # Khởi tạo biến theo dõi trạng thái training
        start_iteration = 0
        t_so_far = 0
        checkpoint_path = os.path.join(save_folder, 'latest_checkpoint.pt')
        
        # Kiểm tra có checkpoint trước đó không
        if resume and os.path.exists(checkpoint_path):
            try:
                print(f"Đang khôi phục từ checkpoint: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                
                # Khôi phục trạng thái training
                start_iteration = checkpoint['iteration']
                t_so_far = checkpoint['t_so_far']
                
                # Khôi phục trạng thái model và optimizer
                for id in range(self.num_agent):
                    self.actors[id].load_state_dict(checkpoint['actors'][id])
                    self.critics[id].load_state_dict(checkpoint['critics'][id])
                    self.optimizers[id].load_state_dict(checkpoint['optimizers'][id])
                    self.loggers[id] = checkpoint['loggers'][id]
                
                print(f"Tiếp tục từ iteration {start_iteration}/{trained_iterations}")
            except Exception as e:
                print(f"Lỗi khi khôi phục checkpoint: {e}")
                print("Bắt đầu training từ đầu")
                start_iteration = 0
                t_so_far = 0
        
        # Chuẩn bị log file
        log_file_path = os.path.join(save_folder, 'log_cur.csv')
        self.file_log_all = open(log_file_path, 'a' if resume and os.path.exists(log_file_path) else 'w', newline='')
        self.writer_log_all = csv.writer(self.file_log_all)
        
        # Khởi tạo TensorBoard writers
        writers = [SummaryWriter(f"runs/ippo/{qq}") for qq in range(self.num_agent)]
        start_time = time.time()
        logs = [[] for _ in range(self.num_agent)]
        i_so_far = start_iteration
        
        try:
            while i_so_far <= trained_iterations:                                           
                batch_states_all, batch_actions_all, batch_log_probs_all, batch_rewards_all, batch_next_states_all, batch_advantages_all, batch_returns_all, batch_values_all = self.roll_out()
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

                    # TRY NOT TO MODIFY: record rewards for plotting purposes
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
                        
                    # Print a summary of our training so far
                    logs[id].append(self._log_summary(id))
                    
                    # Save our model if it's time
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
                
                # Lưu checkpoint mới nhất sau mỗi lần lặp
                self._save_checkpoint(checkpoint_path, i_so_far, t_so_far)
                
        except KeyboardInterrupt:
            print("Training bị dừng bởi người dùng. Lưu checkpoint...")
            self._save_checkpoint(checkpoint_path, i_so_far, t_so_far)
        except Exception as e:
            print(f"Lỗi xảy ra trong training: {e}. Lưu checkpoint...")
            self._save_checkpoint(checkpoint_path, i_so_far, t_so_far)
        finally:
            # Đảm bảo đóng file log
            if self.file_log_all:
                self.file_log_all.close()
            # Đóng TensorBoard writers
            for writer in writers:
                writer.close()
            
        return i_so_far

    def _save_checkpoint(self, path, iteration, t_so_far):
        """Lưu checkpoint với tất cả trạng thái cần thiết để tiếp tục training"""
        checkpoint = {
            'iteration': iteration,
            't_so_far': t_so_far,
            'actors': {id: self.actors[id].state_dict() for id in range(self.num_agent)},
            'critics': {id: self.critics[id].state_dict() for id in range(self.num_agent)},
            'optimizers': {id: self.optimizers[id].state_dict() for id in range(self.num_agent)},
            'loggers': {id: self.loggers[id] for id in range(self.num_agent)}
        }
        
        # Lưu tạm vào file tạm, sau đó đổi tên để tránh mất dữ liệu nếu bị dừng khi đang lưu
        temp_path = f"{path}.tmp"
        torch.save(checkpoint, temp_path)
        if os.path.exists(temp_path):  # Kiểm tra file tạm đã được tạo thành công
            os.replace(temp_path, path)  # Đổi tên file tạm thành file chính
                

                
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
        
