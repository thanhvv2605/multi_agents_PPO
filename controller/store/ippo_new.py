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
import copy
import shutil
import csv

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
            'i_so_far': 0, 't_so_far': 0, 'ep_lens': [], 'ep_rewards': [],
            'losses': [], 'rewards': [], 'delta_t': time.time_ns()
        } for _ in range(self.num_agent)] # 'ep_lifetime' replaced with 'ep_rewards'

        if model_path is not None:
            try:
                agent_folders = [d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))]
                for agent_folder in agent_folders:
                    agent_index = int(agent_folder)
                    if 0 <= agent_index < self.num_agent:
                        agent_path = os.path.join(model_path, agent_folder)
                        actor_path = os.path.join(agent_path, "actor.pth")
                        critic_path = os.path.join(agent_path, "critic.pth")
                        log_path = os.path.join(agent_path, "log.csv")

                        if os.path.exists(actor_path):
                            self.actors[agent_index].load_state_dict(torch.load(actor_path, map_location=self.device))
                        if os.path.exists(critic_path):
                            self.critics[agent_index].load_state_dict(torch.load(critic_path, map_location=self.device))

                        self.actors[agent_index].to(self.device)
                        self.critics[agent_index].to(self.device)

                        if os.path.exists(log_path):
                            self.log_file[agent_index] = log_path
                            with open(log_path, 'r') as file:
                                reader = csv.reader(file)
                                rows = list(reader)
                                if rows:
                                    last_row = rows[-1]
                                    if len(last_row) >= 2:
                                        self.loggers[agent_index]['i_so_far'] = int(last_row[0])
                                        self.loggers[agent_index]['t_so_far'] = int(last_row[1])
            except Exception as e:
                 print(f"Error loading model from {model_path}: {e}. Starting fresh.")
                 self.loggers = [{
                    'i_so_far': 0, 't_so_far': 0, 'ep_lens': [], 'ep_rewards': [],
                    'losses': [], 'rewards': [], 'delta_t': time.time_ns()
                 } for _ in range(self.num_agent)]

        self.optimizers = [None for _ in range(self.num_agent)]
        for i in range(self.num_agent):
            parameters = list(self.actors[i].parameters()) + list(self.critics[i].parameters())
            self.optimizers[i] = optim.Adam(parameters, lr=args["lr"])

    def cal_rt_adv(self, agent_id, states, rewards, next_states, terminals):
        # Assuming terminals is a Tensor where 1 means terminal, 0 means not terminal
        with torch.no_grad():
            values = self.get_value(agent_id, states)
            next_values = self.get_value(agent_id, next_states)
            # Ensure terminals is integer type for masking
            terminals_int = terminals.int()
            non_terminals_mask = 1 - terminals_int

            if self.gae:
                advantages = torch.zeros_like(rewards)
                lastgaelam = 0
                for t in reversed(range(len(rewards))):
                     # Correct GAE: Use (1 - terminal) mask for next_value and next_advantage
                    delta = rewards[t] + self.gamma * next_values[t] * non_terminals_mask[t] - values[t]
                    lastgaelam = delta + self.gamma * self.gae_lambda * non_terminals_mask[t] * lastgaelam
                    advantages[t] = lastgaelam
                returns = advantages + values
            else:
                # Non-GAE Calculation (Simplified)
                returns = torch.zeros_like(rewards)
                next_return = next_values[-1] * non_terminals_mask[-1] # Start with last next_value if not terminal
                for t in reversed(range(len(rewards))):
                    returns[t] = rewards[t] + self.gamma * next_return * non_terminals_mask[t]
                    next_return = returns[t] # Use current return as next for previous step
                advantages = returns - values
            return returns, advantages, values

    def get_action(self, agent_id, state_in):
        state = torch.FloatTensor(np.array(state_in)).to(self.device) # Ensure input is numpy array first
        if state.ndim == 3:
            state = state.unsqueeze(0)
        elif state.ndim < 3:
             # Handle potential dimension issues if state is not as expected
             print(f"Warning: Unexpected state dimension {state.ndim} for agent {agent_id}")
             # Add logic here to reshape or handle appropriately based on your state format
             # Example: state = state.view(1, C, H, W) if needed
             pass # Placeholder

        mean, log_std = self.actors[agent_id](state)
        std = log_std.exp()
        dist = Normal(mean, std)
        action = dist.sample()
        # Sum log_prob across action dimensions (assuming action shape is like [batch, chan, height, width])
        action_log_prob = dist.log_prob(action).sum(dim=tuple(range(1, action.dim())))

        return action.detach().cpu().numpy(), action_log_prob.detach().cpu().numpy()

    def evaluate(self, agent_id, batch_states, batch_actions):
        total_batch = batch_states.size(0)
        sub_batch_size = min(self.minibatch_size, total_batch) # Use minibatch_size for consistency

        action_log_probs = []
        entropies = []

        self.actors[agent_id].eval() # Set actor to evaluation mode
        with torch.no_grad(): # No gradients needed for evaluation metrics
            for start_idx in range(0, total_batch, sub_batch_size):
                end_idx = min(start_idx + sub_batch_size, total_batch)
                sub_states = batch_states[start_idx:end_idx]
                sub_actions = batch_actions[start_idx:end_idx]

                mean, log_std = self.actors[agent_id](sub_states)
                std = torch.exp(log_std)
                dist = Normal(mean, std)
                # Sum log_prob across action dimensions
                sub_log_probs = dist.log_prob(sub_actions).sum(dim=tuple(range(1, sub_actions.dim())))
                # Sum entropy across action dimensions
                sub_entropy = dist.entropy().sum(dim=tuple(range(1, sub_actions.dim())))

                action_log_probs.append(sub_log_probs)
                entropies.append(sub_entropy)
        self.actors[agent_id].train() # Set actor back to training mode

        return torch.cat(action_log_probs), torch.cat(entropies)

    def get_value(self, agent_id, state):
        # Assuming critic output needs summing (like pixel-wise values)
        # If critic outputs a single value per state, remove .sum(1)
        return self.critics[agent_id](state).sum(1)

    def roll_out(self):
        # Initialize lists to hold trajectory data for all agents for the entire batch collection phase
        batch_states_list = [[] for _ in range(self.num_agent)]
        batch_actions_list = [[] for _ in range(self.num_agent)]
        batch_log_probs_list = [[] for _ in range(self.num_agent)]
        batch_next_states_list = [[] for _ in range(self.num_agent)]
        batch_rewards_list = [[] for _ in range(self.num_agent)]
        batch_terminals_list = [[] for _ in range(self.num_agent)] # Store terminals too

        # Track collected samples per agent
        t_collected = [0 for _ in range(self.num_agent)]
        all_agents_ready = False

        while not all_agents_ready:
            # Per-episode data storage
            ep_states = [[] for _ in range(self.num_agent)]
            ep_actions = [[] for _ in range(self.num_agent)]
            ep_log_probs = [[] for _ in range(self.num_agent)]
            ep_next_states = [[] for _ in range(self.num_agent)]
            ep_rewards = [[] for _ in range(self.num_agent)]
            ep_terminals = [[] for _ in range(self.num_agent)]

            request = self.env.reset()
            ep_len_count = [0 for _ in range(self.num_agent)]
            ep_log_probs_pre = [None for _ in range(self.num_agent)]
            ep_done = False
            print("Sample count ", t_collected)

            while not ep_done:
                agent_id = request["agent_id"]
                state = request["state"]
                action, log_prob = self.get_action(agent_id, state)
                ep_log_probs_pre[agent_id] = log_prob

                request = self.env.step(agent_id, action)
                next_state = request["state"]
                reward = request["reward"]
                terminal = request["terminal"] # Assume 1 if terminal, 0 otherwise
                prev_state = request["prev_state"]
                input_action = request["input_action"] # Action that was input to env

                # Store transition only if the agent has taken an action before in this ep
                if ep_log_probs_pre[agent_id] is not None:
                     # Check if we still need samples for this agent
                    if t_collected[agent_id] < self.batch_size:
                        ep_states[agent_id].append(prev_state)
                        ep_actions[agent_id].append(input_action) # Store the actual action taken
                        ep_log_probs[agent_id].append(ep_log_probs_pre[agent_id])
                        ep_next_states[agent_id].append(next_state)
                        ep_rewards[agent_id].append(reward)
                        ep_terminals[agent_id].append(terminal)
                        ep_len_count[agent_id] += 1
                        t_collected[agent_id] += 1

                # Check if the episode has finished for the environment
                if terminal: # Assuming terminal applies to the whole episode context
                    ep_done = True
                    break


                # Check if all agents have enough samples after this step
                all_agents_ready = all(t >= self.batch_size for t in t_collected)
                if all_agents_ready:
                     ep_done = True # Stop current episode collection early if batch is full


            # Process finished episode data
            for agent_id in range(self.num_agent):
                if len(ep_states[agent_id]) > 0:
                    batch_states_list[agent_id].extend(ep_states[agent_id])
                    batch_actions_list[agent_id].extend(ep_actions[agent_id])
                    batch_log_probs_list[agent_id].extend(ep_log_probs[agent_id])
                    batch_next_states_list[agent_id].extend(ep_next_states[agent_id])
                    batch_rewards_list[agent_id].extend(ep_rewards[agent_id])
                    batch_terminals_list[agent_id].extend(ep_terminals[agent_id])

                    self.loggers[agent_id]["ep_lens"].append(ep_len_count[agent_id])
                    self.loggers[agent_id]["ep_rewards"].append(np.sum(ep_rewards[agent_id])) # Log total reward for this episode


        # --- Calculate Advantages and Returns for collected batches ---
        final_batch_states = [None] * self.num_agent
        final_batch_actions = [None] * self.num_agent
        final_batch_log_probs = [None] * self.num_agent
        final_batch_advantages = [None] * self.num_agent
        final_batch_returns = [None] * self.num_agent
        final_batch_values = [None] * self.num_agent

        for agent_id in range(self.num_agent):
            if not batch_states_list[agent_id]:
                continue

            num_collected = len(batch_states_list[agent_id])
            indices = np.arange(num_collected) # Default to using all collected initially

            if num_collected > self.batch_size:
                # If collected more than batch_size, randomly sample batch_size indices
                indices = np.random.choice(num_collected, self.batch_size, replace=False)
            elif num_collected < self.batch_size:
                print(f"Warning: Agent {agent_id} collected {num_collected} < {self.batch_size} samples.")
                # Use all available samples

            # Select data using indices
            selected_states = np.array(batch_states_list[agent_id])[indices]
            selected_actions = np.array(batch_actions_list[agent_id])[indices]
            selected_log_probs = np.array(batch_log_probs_list[agent_id])[indices]
            selected_rewards = np.array(batch_rewards_list[agent_id])[indices]
            selected_next_states = np.array(batch_next_states_list[agent_id])[indices]
            selected_terminals = np.array(batch_terminals_list[agent_id])[indices]

            # Convert selected data to Tensors
            states_tensor = torch.FloatTensor(selected_states).to(self.device)
            actions_tensor = torch.FloatTensor(selected_actions).to(self.device)
            log_probs_tensor = torch.FloatTensor(selected_log_probs).to(self.device)
            rewards_tensor = torch.FloatTensor(selected_rewards).to(self.device)
            next_states_tensor = torch.FloatTensor(selected_next_states).to(self.device)
            terminals_tensor = torch.FloatTensor(selected_terminals).to(self.device)

            # Calculate returns, advantages, and values using tensors
            returns, advantages, values = self.cal_rt_adv(
                agent_id, states_tensor, rewards_tensor, next_states_tensor, terminals_tensor
            )

            # Store final tensors
            final_batch_states[agent_id] = states_tensor
            final_batch_actions[agent_id] = actions_tensor
            final_batch_log_probs[agent_id] = log_probs_tensor
            final_batch_advantages[agent_id] = advantages
            final_batch_returns[agent_id] = returns
            final_batch_values[agent_id] = values # Values computed within cal_rt_adv


        # Return tuple of lists, where each list contains tensors for each agent
        return (final_batch_states, final_batch_actions, final_batch_log_probs,
                final_batch_advantages, final_batch_returns, final_batch_values)


    def train(self, trained_iterations, save_folder, resume=True):
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        start_iteration = 0
        global_t_so_far = 0 # Use a single global step counter if desired
        checkpoint_path = os.path.join(save_folder, 'latest_checkpoint.pt')

        if resume and os.path.exists(checkpoint_path):
            try:
                print(f"Resuming from checkpoint: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                start_iteration = checkpoint.get('iteration', 0)
                global_t_so_far = checkpoint.get('t_so_far', 0)

                for agent_id in range(self.num_agent):
                    if 'actors' in checkpoint and agent_id in checkpoint['actors']:
                        self.actors[agent_id].load_state_dict(checkpoint['actors'][agent_id])
                    if 'critics' in checkpoint and agent_id in checkpoint['critics']:
                        self.critics[agent_id].load_state_dict(checkpoint['critics'][agent_id])
                    if 'optimizers' in checkpoint and agent_id in checkpoint['optimizers']:
                         self.optimizers[agent_id].load_state_dict(checkpoint['optimizers'][agent_id])
                    if 'loggers' in checkpoint and agent_id in checkpoint['loggers']:
                         self.loggers[agent_id] = checkpoint['loggers'][agent_id]
                    # Ensure models are on the correct device after loading
                    self.actors[agent_id].to(self.device)
                    self.critics[agent_id].to(self.device)

                print(f"Resumed from iteration {start_iteration}/{trained_iterations}")
            except Exception as e:
                print(f"Error resuming checkpoint: {e}. Starting fresh.")
                start_iteration = 0
                global_t_so_far = 0

        writers = [SummaryWriter(os.path.join("runs", save_folder, f"agent_{i}")) for i in range(self.num_agent)]
        overall_writer = SummaryWriter(os.path.join("runs", save_folder, "overall")) # Writer for aggregated stats
        start_time = time.time()

        print(f"Starting training from iteration {start_iteration + 1}")
        i_so_far = start_iteration

        try:
            while i_so_far < trained_iterations:
                i_so_far += 1
                iter_start_time = time.time_ns()

                # Rollout: Collect data for all agents up to batch_size
                batch_data = self.roll_out()
                (batch_states_all, batch_actions_all, batch_log_probs_all,
                 batch_advantages_all, batch_returns_all, batch_values_all) = batch_data

                current_batch_timesteps = sum(len(b) for b in batch_states_all if b is not None)
                global_t_so_far += current_batch_timesteps

                logs_this_iter = [] # Store summaries for aggregation

                # Update each agent
                for agent_id in range(self.num_agent):
                    # Skip if agent collected no data (shouldn't happen with current rollout)
                    if batch_states_all[agent_id] is None:
                        continue

                    # Extract data for the current agent
                    batch_states = batch_states_all[agent_id]
                    batch_actions = batch_actions_all[agent_id]
                    batch_log_probs = batch_log_probs_all[agent_id]
                    batch_advantages = batch_advantages_all[agent_id]
                    batch_returns = batch_returns_all[agent_id]
                    batch_values = batch_values_all[agent_id]

                    current_agent_batch_size = batch_states.size(0)
                    if current_agent_batch_size == 0: continue # Skip if batch is empty

                    self.loggers[agent_id]['i_so_far'] = i_so_far
                    self.loggers[agent_id]['t_so_far'] = global_t_so_far # Use global counter

                    b_inds = np.arange(current_agent_batch_size)
                    clipfracs = []
                    pg_losses, v_losses, ent_losses = [], [], []
                    approx_kls, old_approx_kls = [], []

                    self.actors[agent_id].train()
                    self.critics[agent_id].train()

                    for _ in range(self.n_updates_per_iteration):
                        np.random.shuffle(b_inds)
                        for start in range(0, current_agent_batch_size, self.minibatch_size):
                            end = min(start + self.minibatch_size, current_agent_batch_size)
                            mb_inds = b_inds[start:end]

                            # Ensure batch_log_probs is correctly indexed and shaped
                            mb_log_probs_old = batch_log_probs[mb_inds]

                            newlogprob, entropy = self.evaluate(agent_id, batch_states[mb_inds], batch_actions[mb_inds])
                            newvalue = self.get_value(agent_id, batch_states[mb_inds]) # Removed squeeze

                            logratio = newlogprob - mb_log_probs_old
                            ratio = logratio.exp()

                            with torch.no_grad():
                                old_approx_kl = (-logratio).mean()
                                approx_kl = ((ratio - 1) - logratio).mean()
                                clipfrac = ((ratio - 1.0).abs() > self.clip).float().mean().item()

                            mb_advantages = batch_advantages[mb_inds]
                            if self.norm_adv:
                                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                            # Policy loss
                            pg_loss1 = -mb_advantages * ratio
                            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip, 1 + self.clip)
                            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                            # Value loss
                            mb_returns = batch_returns[mb_inds]
                            newvalue = newvalue.view(-1) # Ensure correct shape
                            if self.clip_vloss:
                                v_loss_unclipped = (newvalue - mb_returns) ** 2
                                v_clipped = batch_values[mb_inds] + torch.clamp(
                                    newvalue - batch_values[mb_inds], -self.clip, self.clip,
                                )
                                v_loss_clipped = (v_clipped - mb_returns) ** 2
                                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                            else:
                                v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

                            entropy_loss = entropy.mean()
                            loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                            self.optimizers[agent_id].zero_grad()
                            loss.backward()
                            nn.utils.clip_grad_norm_(self.actors[agent_id].parameters(), self.max_grad_norm)
                            nn.utils.clip_grad_norm_(self.critics[agent_id].parameters(), self.max_grad_norm)
                            self.optimizers[agent_id].step()

                            # Store minibatch stats
                            pg_losses.append(pg_loss.item())
                            v_losses.append(v_loss.item())
                            ent_losses.append(entropy_loss.item())
                            approx_kls.append(approx_kl.item())
                            old_approx_kls.append(old_approx_kl.item())
                            clipfracs.append(clipfrac)
                            self.loggers[agent_id]['losses'].append(loss.item())


                    # Log training stats for this agent to TensorBoard
                    y_pred = batch_values.cpu().numpy()
                    y_true = batch_returns.cpu().numpy()
                    var_y = np.var(y_true)
                    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

                    writers[agent_id].add_scalar("losses/policy_loss", np.mean(pg_losses), i_so_far)
                    writers[agent_id].add_scalar("losses/value_loss", np.mean(v_losses), i_so_far)
                    writers[agent_id].add_scalar("losses/entropy", np.mean(ent_losses), i_so_far)
                    writers[agent_id].add_scalar("losses/approx_kl", np.mean(approx_kls), i_so_far)
                    writers[agent_id].add_scalar("losses/old_approx_kl", np.mean(old_approx_kls), i_so_far)
                    writers[agent_id].add_scalar("losses/clipfrac", np.mean(clipfracs), i_so_far)
                    writers[agent_id].add_scalar("losses/explained_variance", explained_var, i_so_far)
                    if self.loggers[agent_id]["ep_rewards"]: # Check if list is not empty
                        writers[agent_id].add_scalar("rollout/ep_rew_mean", np.mean(self.loggers[agent_id]["ep_rewards"]), i_so_far)
                    if self.loggers[agent_id]["ep_lens"]: # Check if list is not empty
                        writers[agent_id].add_scalar("rollout/ep_len_mean", np.mean(self.loggers[agent_id]["ep_lens"]), i_so_far)

                    # Compute summary stats for this agent (doesn't print here)
                    log_data_agent = self._compute_and_reset_log(agent_id)
                    if log_data_agent:
                         logs_this_iter.append(log_data_agent)

                    # Save model periodically
                    if i_so_far % self.save_freq == 0:
                         self._save_model(agent_id, i_so_far, save_folder)


                # --- Aggregated Logging and Checkpointing after processing all agents ---
                iter_end_time = time.time_ns()
                iter_duration = (iter_end_time - iter_start_time) / 1e9
                sps = int(current_batch_timesteps / iter_duration) if iter_duration > 0 else 0

                if logs_this_iter:
                    avg_all_ep_lens = np.mean([log['avg_ep_lens'] for log in logs_this_iter])
                    avg_all_ep_return = np.mean([log['avg_ep_return'] for log in logs_this_iter])
                    avg_all_loss = np.mean([log['avg_loss'] for log in logs_this_iter])

                    print(flush=True)
                    print(f"--- Iteration #{i_so_far} Summary (Avg across {len(logs_this_iter)} agents) ---", flush=True)
                    print(f"Avg Episodic Length: {avg_all_ep_lens:.2f}", flush=True)
                    print(f"Avg Episodic Return: {avg_all_ep_return:.2f}", flush=True)
                    print(f"Avg Loss: {avg_all_loss:.5f}", flush=True)
                    print(f"Timesteps So Far: {global_t_so_far}", flush=True)
                    print(f"Iteration Took: {iter_duration:.2f} secs (SPS: {sps})", flush=True)
                    print(f"-----------------------------------------------------", flush=True)
                    print(flush=True)

                    # Log aggregated stats to overall writer
                    overall_writer.add_scalar("charts/SPS", sps, i_so_far)
                    overall_writer.add_scalar("rollout/avg_ep_len", avg_all_ep_lens, i_so_far)
                    overall_writer.add_scalar("rollout/avg_ep_return", avg_all_ep_return, i_so_far)
                    overall_writer.add_scalar("losses/avg_loss", avg_all_loss, i_so_far)
                    overall_writer.add_scalar("charts/learning_rate", self.optimizers[0].param_groups[0]["lr"], i_so_far) # Log one LR

                # Save latest checkpoint
                self._save_checkpoint(checkpoint_path, i_so_far, global_t_so_far)

        except KeyboardInterrupt:
            print("\nTraining interrupted by user. Saving final checkpoint...")
        except Exception as e:
             import traceback
             print(f"\nAn error occurred during training: {e}")
             print(traceback.format_exc())
             print("Saving checkpoint before exiting...")
        finally:
            self._save_checkpoint(checkpoint_path, i_so_far, global_t_so_far) # Ensure save on exit
            print("Closing writers...")
            for writer in writers:
                 writer.close()
            overall_writer.close()
            print("Training finished.")

        return i_so_far

    def _save_model(self, agent_id, iteration, save_folder):
        """Saves individual agent model and potentially log file."""
        folder = os.path.join(save_folder, str(iteration), str(agent_id))
        os.makedirs(folder, exist_ok=True)
        torch.save(self.actors[agent_id].state_dict(), os.path.join(folder, 'actor.pth'))
        torch.save(self.critics[agent_id].state_dict(), os.path.join(folder, 'critic.pth'))

        # Optional: Save agent-specific log history if needed
        # log_hist_path = os.path.join(folder, 'log_history.csv')
        # current_log_file = self.log_file[agent_id]
        # if current_log_file and os.path.exists(current_log_file):
        #     try: shutil.copy(current_log_file, log_hist_path)
        #     except Exception as e: print(f"Could not copy log file {current_log_file}: {e}")


    def _save_checkpoint(self, path, iteration, t_so_far):
        """Saves training state for resuming."""
        checkpoint = {
            'iteration': iteration,
            't_so_far': t_so_far,
            'actors': {id: self.actors[id].state_dict() for id in range(self.num_agent)},
            'critics': {id: self.critics[id].state_dict() for id in range(self.num_agent)},
            'optimizers': {id: self.optimizers[id].state_dict() for id in range(self.num_agent)},
            'loggers': {id: self.loggers[id] for id in range(self.num_agent)} # Save logger state too
        }
        temp_path = f"{path}.tmp"
        try:
            torch.save(checkpoint, temp_path)
            os.replace(temp_path, path)
            # print(f"Checkpoint saved to {path} at iteration {iteration}") # Optional: verbose saving log
        except Exception as e:
            print(f"Error saving checkpoint to {path}: {e}")
        finally:
            if os.path.exists(temp_path):
                try: os.remove(temp_path) # Clean up temp file if rename failed
                except OSError: pass


    def _compute_and_reset_log(self, agent_id):
        """Computes averages for the completed iteration and resets logger lists."""
        logger = self.loggers[agent_id]
        if not logger['ep_lens']: # No episodes completed for this agent in this rollout
             # Reset lists anyway
             logger['ep_lens'] = []
             logger['ep_rewards'] = []
             logger['losses'] = []
             logger['rewards'] = [] # Ensure rewards are reset
             return None

        avg_ep_lens = np.mean(logger['ep_lens'])
        avg_ep_return = np.mean(logger['ep_rewards']) # Use the stored episodic rewards
        avg_loss = np.mean(logger['losses']) if logger['losses'] else 0

        # Reset batch-specific logging data
        logger['ep_lens'] = []
        logger['ep_rewards'] = []
        logger['losses'] = []
        logger['rewards'] = [] # Make sure rewards collected per step are also reset

        return {'avg_ep_lens': avg_ep_lens, 'avg_ep_return': avg_ep_return, 'avg_loss': avg_loss}