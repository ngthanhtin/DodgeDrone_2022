"""This is OpenAI' Spinning Up PyTorch implementation of Soft-Actor-Critic with
minor adjustments.
For the official documentation, see below:
https://spinningup.openai.com/en/latest/algorithms/sac.html#documentation-pytorch-version
Source:
https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/sac.py
"""
import itertools
import queue, threading
from copy import deepcopy
from tkinter import E
import cv2

import torch
import numpy as np
from gym.spaces import Box, Discrete
from torch.optim import Adam, SGD

from agents.base import BaseAgent
from l2f.common.models.network import ActorCritic
from l2f.common.models.vae import VAE
from l2f.common.utils import setup_logging

from ruamel.yaml import YAML

from agents.replay_buffer import ReplayBuffer

from stable_baselines3.common.utils import get_device

DEVICE = get_device('auto')

seed = np.random.randint(255)
torch.manual_seed(seed)
np.random.seed(seed)


class SACAgent(BaseAgent):
    """Adopted from https://github.com/learn-to-race/l2r/blob/main/l2r/baselines/rl/sac.py"""

    def __init__(self, num_envs):
        super(SACAgent, self).__init__()

        self.cfg = self.load_model_config("./python2/models/sac/params-sac.yaml")
        self.file_logger, self.tb_logger = self.setup_loggers()

        self.setup_vision_encoder()
        self.num_envs = num_envs
        self.set_params()

        self.frame_id = 0 # use for get rgb and depth images

    def select_action(self, obs, encode=True):
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy.
        if encode:
            obs = self._encode(obs)
        if self.t > self.cfg["start_steps"]:
            a = self.actor_critic.act(obs.to(DEVICE), self.deterministic)
            self.record["transition_actor"] = "learner"
        else:
            a = self.action_space.sample()
            # action_space = Box(-1, 1, (self.num_envs, 4), dtype=np.float64)
            # a = action_space.sample()
            self.record["transition_actor"] = "random"
        self.t = self.t + 1
        
        # a = np.expand_dims(a, 0) # to be compatible with envs
        a = a.astype(np.float64) # to be compatible with envs

        return a

    def register_reset(self, obs) -> np.array:
        """
        Same input/output as select_action, except this method is called at episodal reset.
        """
        # camera, features, state = obs
        self.deterministic = True
        self.t = 1e6

    def load_model(self, path):
        self.actor_critic.load_state_dict(torch.load(path))

    def save_model(self, path):
        torch.save(self.actor_critic.state_dict(), path)

    def setup_vision_encoder(self):
        assert self.cfg["use_encoder_type"] in [
            "vae"
        ], "Specified encoder type must be in ['vae']"
        state_hiddens = self.cfg[self.cfg["use_encoder_type"]]["state_hiddens"]
        self.feat_dim = self.cfg[self.cfg["use_encoder_type"]]["latent_dims"] + 55
        self.obs_dim = (
            self.cfg[self.cfg["use_encoder_type"]]["latent_dims"] + state_hiddens[-1]
            if self.cfg["encoder_switch"]
            else None
        )

        if self.cfg["use_encoder_type"] == "vae":
            self.backbone = VAE(
                im_c=self.cfg["vae"]["im_c"],
                im_h=self.cfg["vae"]["im_h"],
                im_w=self.cfg["vae"]["im_w"],
                z_dim=self.cfg["vae"]["latent_dims"],
            )
            # self.backbone.load_state_dict(
            #     torch.load(self.cfg["vae"]["vae_chkpt_statedict"], map_location=DEVICE)
            # )
        else:
            raise NotImplementedError

        self.backbone.to(DEVICE)

    def set_params(self):
        self.save_episodes = True
        self.episode_num = 0
        self.best_ret = -999
        self.t = 0
        self.deterministic = False
        self.pi_scheduler = None
        self.t_start = 0

        # This is important: it allows child classes (that extend this one) to "push up" information
        # that this parent class should log
        self.metadata = {}
        self.record = {"transition_actor": ""}

        # self.action_space = Box(-1, 1, (4, ))
        self.action_space = Box(-1, 1, (self.num_envs, 4), dtype=np.float64)
        # self.act_dim = self.action_space.shape[0]
        self.act_dim = self.action_space.shape[1]
        
        # Experience buffer
        self.replay_buffer = ReplayBuffer(
            num_envs=self.num_envs, obs_dim=self.feat_dim, act_dim=self.act_dim, size=self.cfg["replay_size"]
        )
        
        self.actor_critic = ActorCritic(
            self.obs_dim,
            self.action_space,
            self.cfg,
            latent_dims=self.obs_dim,
            device=DEVICE,
        )

        if self.cfg["checkpoint"] and self.cfg["load_checkpoint"]:
            self.load_model(self.cfg["checkpoint"])

        self.actor_critic_target = deepcopy(self.actor_critic)

    @staticmethod
    def load_model_config(path):
        yaml = YAML()
        params = yaml.load(open(path))
        sac_kwargs = params["agent_kwargs"]
        return sac_kwargs

    def setup_loggers(self):
        save_path = self.cfg["model_save_path"]
        loggers = setup_logging(save_path, self.cfg["experiment_name"], True)
        loggers[0]("Using random seed: {}".format(0))
        return loggers

    def compute_loss_q(self, data):
        """Set up function for computing SAC Q-losses."""
        o, a, r, o2, d = (
            data["obs"],
            data["act"],
            data["rew"],
            data["obs2"],
            data["done"],
        )
        
        q1 = self.actor_critic.q1(o, a)
        q2 = self.actor_critic.q2(o, a)
        
        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.actor_critic.pi(o2)
            # Target Q-values
            q1_pi_targ = self.actor_critic_target.q1(o2, a2)
            q2_pi_targ = self.actor_critic_target.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            
            backup = r + self.cfg["gamma"] * (1 - d) * (
                q_pi_targ - self.cfg["alpha"] * logp_a2
            )

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()

        loss_q = loss_q1 + loss_q2
        # Useful info for logging
        q_info = dict(
            Q1Vals=q1.detach().cpu().numpy(), Q2Vals=q2.detach().cpu().numpy()
        )

        return loss_q, q_info

    def compute_loss_pi(self, data):
        """Set up function for computing SAC pi loss."""
        o = data["obs"]
        pi, logp_pi = self.actor_critic.pi(o)
        q1_pi = self.actor_critic.q1(o, pi)
        q2_pi = self.actor_critic.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.cfg["alpha"] * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().cpu().numpy())

        return loss_pi, pi_info

    def update(self, data):
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q, q_info = self.compute_loss_q(data)
        loss_q.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 40)
        torch.nn.utils.clip_grad_norm_(self.actor_critic_target.parameters(), 40)
        self.q_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()
        self.pi_scheduler.step()
        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(
                self.actor_critic.parameters(), self.actor_critic_target.parameters()
            ):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.cfg["polyak"])
                p_targ.data.add_((1 - self.cfg["polyak"]) * p.data)

        return pi_info, q_info

    def _step(self, env, action):
        obs, reward, done, info = env.step(action)
        
        #update frame id
        receive_frame_id = env.render(frame_id = self.frame_id)
        # print("sending frame id: ", self.frame_id, "received frame id: ", receive_frame_id)
        self.frame_id+=1
        rgb_img, depth_img = self.get_RGB_and_Depth(env)
        camera = rgb_img
        
        # return image, features (vae), state, reward, done, info
        return camera, self._encode((obs, camera)), obs, reward, done, info

    def get_RGB_and_Depth(self, env):
        raw_rgb_img = env.getImage(rgb=True) 

        num_img = raw_rgb_img.shape[0] 
        num_col = 1
        num_row = int(num_img / num_col)

        rgb_img_list = []
        for col in range(num_col):
            rgb_img_list.append([])
            for row in range(num_row):
                rgb_img = np.reshape(
                    raw_rgb_img[col*num_row + row], (env.img_height, env.img_width, 3))
                rgb_img_list[col] += [rgb_img]
        # rgb_img_tile = cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in rgb_img_list]) #(240, 320, 3)
        rgb_img_tile = np.stack(np.array(rgb_img_list))
        rgb_img_tile = np.squeeze(rgb_img_tile)
        rgb_img_tile = rgb_img_tile

        # depth image
        raw_depth_images = env.getDepthImage()
        depth_img_list = []
        for col in range(num_col):
            depth_img_list.append([])
            for row in range(num_row):
                depth_img = np.reshape(
                    raw_depth_images[col*num_row + row], (env.img_height, env.img_width))
                depth_img_list[col] += [depth_img]
        # depth_img_tile = cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in depth_img_list])
        depth_img_tile = np.stack(np.array(depth_img_list))
        depth_img_tile = np.squeeze(depth_img_tile)
        depth_img_tile = np.expand_dims(depth_img_tile, -1)
        # depth_img_tile = depth_img_tile*255.
        
        return rgb_img_tile, depth_img_tile

    def _reset(self, env, random_pos=False):
        obs = env.reset(random=random_pos)
        state = obs
        # self.frame_id = 0 # use for get rgb and depth images
        # ====== Retrive RGB and Depth Images From the simulator=========
        receive_frame_id = env.render(frame_id = self.frame_id)
        # print("sending frame id: ", self.frame_id, "received frame id: ", receive_frame_id)
        self.frame_id+=1
        rgb_img, depth_img = self.get_RGB_and_Depth(env)
        camera = rgb_img

        return camera, self._encode((state, camera)), state

    def _encode(self, o):
        state, img = o # img shape: (240, 320, 3)
        if self.num_envs == 1:
            img = np.expand_dims(img, 0)
            
        if self.cfg["use_encoder_type"] == "vae":
            # img_embed = self.backbone.encode_raw(img, DEVICE)[0][0]
            img_embed = self.backbone.encode_raw(img, DEVICE)[0]
            
            # state = (torch.tensor((state)).float().reshape(1, -1).to(DEVICE))
            state = (torch.tensor((state)).float().to(DEVICE))
            if len(state.size()) == 1: # handle if num_envs = 1
                state = state.unsqueeze(0)
            # out = torch.cat([img_embed.unsqueeze(0), state], dim=-1).squeeze(0)  # torch.Size([32])
            out = torch.cat([img_embed, state], dim=-1)
            
            self.using_state = 1
        else:
            raise NotImplementedError

        assert not torch.sum(torch.isnan(out)), "found a nan value"
        out[torch.isnan(out)] = 0

        return out

    def eval(self, n_eps, env):
        print("Evaluation:")
        val_ep_rets = []

        # Not implemented for logging multiple test episodes
        assert self.cfg["num_test_episodes"] == 1

        for j in range(self.cfg["num_test_episodes"]):
            camera, features, state = self._reset(env, random_pos=False)
            a = self.select_action(features, encode=False)
            d, ep_ret, ep_len, n_val_steps, self.metadata = False, 0, 0, 0, {}
            camera, features, state2, r, d, info = self._step(env, a)
            t = 0
            
            while (len(np.nonzero(d)[0]) < 4) & (ep_len <= self.cfg["max_ep_len"]):
                # Take deterministic actions at test time
                self.deterministic = True
                self.t = 1e6
                a = self.select_action(features, encode=False)
                camera2, features2, state2, r, d, info = self._step(env, a)
                
                ep_ret += sum(r)/len(r)
            
                ep_len += self.num_envs
                n_val_steps += 1

                features = features2
                camera = camera2
                state = state2
                t += 1

            self.file_logger(f"[eval episode] {info}")

            val_ep_rets.append(ep_ret)
            self.metadata["info"] = info
            test_r = [v["episode"]["r"] for v in self.metadata["info"] if v != {}]
            if sum(test_r) == 0:
                continue
            self.log_val_metrics_to_tensorboard(info, ep_ret, n_eps, n_val_steps)

        self.deterministic = False
        self.checkpoint_model(ep_ret, n_eps)

        return val_ep_rets

    def checkpoint_model(self, ep_ret, n_eps):
        print("N eps: ", n_eps)
        # Save if best (or periodically)
        mean_ep_ret = ep_ret
        if mean_ep_ret > self.best_ret:  # and ep_ret > 100):
            path_name = f"{self.cfg['model_save_path']}/best_{self.cfg['experiment_name']}_episode_{n_eps}.statedict"
            self.file_logger(
                f"New best episode reward of {round(mean_ep_ret, 1)}! Saving: {path_name}"
            )
            self.best_ret = mean_ep_ret
            torch.save(self.actor_critic.state_dict(), path_name)
            path_name = f"{self.cfg['model_save_path']}/best_{self.cfg['experiment_name']}_episode_{n_eps}.statedict"

        elif self.save_episodes and (n_eps % self.cfg["save_freq"] == 0):
            path_name = f"{self.cfg['model_save_path']}/{self.cfg['experiment_name']}_episode_{n_eps}.statedict"
            self.file_logger(
                f"Periodic save (save_freq of {self.cfg['save_freq']}) to {path_name}"
            )
            torch.save(self.actor_critic.state_dict(), path_name)
            

    def training(self, env):
        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(
            self.actor_critic.q1.parameters(), self.actor_critic.q2.parameters()
        )

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(
            self.actor_critic.policy.parameters(), lr=self.cfg["lr"]
        )
        self.q_optimizer = Adam(self.q_params, lr=self.cfg["lr"])

        # self.pi_optimizer = SGD(self.actor_critic.policy.parameters(), lr=self.cfg["lr"])
        # self.q_optimizer = SGD(self.q_params, lr=self.cfg["lr"])

        self.pi_scheduler = torch.optim.lr_scheduler.StepLR(
            self.pi_optimizer, 1, gamma=0.5
        )

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.actor_critic_target.parameters():
            p.requires_grad = False

        # Prepare for interaction with environment
        best_ret, ep_ret, ep_len = -999, 0, 0
        self.pi_infos, self.q_infos = [], []
        self._reset(env, random_pos=True)
        a = self.action_space.sample()
        
        camera, feat, state, r, d, info = self._step(env, a)
        
        state_dim = 55 if self.using_state else 0
        assert (
            len(feat[0])
            == self.cfg[self.cfg["use_encoder_type"]]["latent_dims"] + state_dim
        ), "'o' has unexpected dimension or is a tuple"

        t_start = self.t_start
        # Main loop: collect experience in env and update/log each epoch
        
        for t in range(self.t_start, self.cfg["total_steps"]):
            a = self.select_action(feat, encode=False)
            # Step the env
            camera2, feat2, state2, r, d, info = self._step(env, a)
            ep_ret += sum(r)/len(r)
            ep_len += self.num_envs
            
            # make training episodic and restart environment on done state
            # if ep_len >= self.cfg["max_ep_len"]:
            #     d = np.array([True]*self.num_envs)
            # d = False if ep_len == self.cfg["max_ep_len"] else d

            # Store experience to replay buffer
            self.replay_buffer.store(feat, a, r, feat2, d)

            feat = feat2
            state = state2
            camera = camera2

            # Update handling
            if (t >= self.cfg["update_after"]) & (t % self.cfg["update_every"] == 0):
                # for j in range(self.cfg["update_every"]):
                for i in range(1):
                    batch = self.replay_buffer.sample_batch(self.cfg["batch_size"])
                    pi_info, q_info = self.update(data=batch)
                self.log_loss_to_tensorboard(pi_info['LogPi'], q_info['Q1Vals'], q_info['Q2Vals'], t)

            if (t + 1) % self.cfg["eval_every"] == 0:
                if best_ret < ep_ret:
                    best_ret = ep_ret
                    self.checkpoint_model(ep_ret, t // self.cfg["eval_every"])

                # # eval on test environment
                # val_returns = self.eval(t // self.cfg["eval_every"], env)

                # Reset
                (
                    camera,
                    ep_len,
                    ep_ret,
                    experience,
                    feat,
                    state,
                    t_start,
                ) = self.reset_each_env(env, feat, t)

            # End of trajectory handling
            if (len(np.nonzero(d)[0]) >= 1) or (ep_len == self.cfg["max_ep_len"]):
            # if (ep_len >= self.cfg["max_ep_len"]):
            # if ep_len >= 5000:
                ep_len = 0
                self.metadata["info"] = [f for f in info if f != {}]
                self.episode_num += 1
                msg = f"[Ep {self.episode_num }] {self.metadata}"
                print("-----------")
                self.file_logger(msg)
                print("-----------")
                test_r = [v["episode"]["r"] for v in self.metadata["info"] if v != {}]
                if sum(test_r) == 0:
                    continue
                self.log_train_metrics_to_tensorboard(ep_ret, t, t_start)

                # Reset
                (
                    camera,
                    ep_len,
                    ep_ret,
                    experience,
                    feat,
                    state,
                    t_start,
                ) = self.reset_each_env(env, feat, t)#self.reset_episode(env, t)

    def reset_each_env(self, env, feat, t):
        action = self.select_action(feat, encode=False)
        ep_ret, ep_len, self.metadata, experience = 0, 0, {}, []
        t_start = t + 1
        camera, feat, state, r, d, info = self._step(env, action)
        return camera, ep_len, ep_ret, experience, feat, state, t_start

    def reset_episode(self, env, t):
        camera, feat, state = self._reset(env, random_pos=True)
        ep_ret, ep_len, self.metadata, experience = 0, 0, {}, []
        t_start = t + 1
        action = self.select_action(feat, encode=False)
        camera, feat, state2, r, d, info = self._step(env, action)
        return camera, ep_len, ep_ret, experience, feat, state, t_start

    
    def log_loss_to_tensorboard(self, pi_loss, q1_loss, q2_loss, n_eps):
        pi_loss = np.mean(pi_loss)
        q1_loss = np.mean(q1_loss)
        q2_loss = np.mean(q2_loss)
        self.tb_logger.add_scalar("loss/pi_loss", pi_loss, n_eps)
        self.tb_logger.add_scalar("loss/q1_loss", q1_loss, n_eps)
        self.tb_logger.add_scalar("loss/q2_loss", q2_loss, n_eps)

    def log_val_metrics_to_tensorboard(self, info, ep_ret, n_eps, n_val_steps):
        self.tb_logger.add_scalar("val/episodic_return", ep_ret, n_eps)
        self.tb_logger.add_scalar("val/ep_n_steps", n_val_steps, n_eps)
        r = [v["episode"]["r"] for v in self.metadata["info"] if v != {}]
        l = [v["episode"]["l"] for v in self.metadata["info"] if v != {}]
        lin_vel_reward = [v["episode"]["lin_vel_reward"] for v in self.metadata["info"] if v != {}]
        collision_penalty = [v["episode"]["collision_penalty"] for v in self.metadata["info"] if v != {}]
        ang_vel_penalty = [v["episode"]["ang_vel_penalty"] for v in self.metadata["info"] if v != {}]
        survive_rew = [v["episode"]["survive_rew"] for v in self.metadata["info"] if v != {}]
        # -----
        r_mean = sum(r) / len(r)
        l_mean = sum(l) / len(l)
        lin_vel_reward_mean = sum(lin_vel_reward) / len(lin_vel_reward)
        collision_penalty_mean = sum(collision_penalty) / len(collision_penalty)
        ang_vel_penalty_mean = sum(ang_vel_penalty) / len(ang_vel_penalty)
        survive_rew_mean = sum(survive_rew) / len(survive_rew)

        self.tb_logger.add_scalar(
            "val/ep_total_rewards",
            r_mean,
            self.episode_num,
        )
        self.tb_logger.add_scalar(
            "val/ep_length",
            l_mean,
            self.episode_num,
        )
        self.tb_logger.add_scalar(
            "val/ep_lin_vel_reward",
            lin_vel_reward_mean,
            self.episode_num,
        )
        self.tb_logger.add_scalar(
            "val/ep_collision_penalty",
            collision_penalty_mean,
            self.episode_num,
        )
        self.tb_logger.add_scalar(
            "val/ep_ang_vel_penalty",
            ang_vel_penalty_mean,
            self.episode_num,
        )
        self.tb_logger.add_scalar(
            "val/ep_survive_rew",
            survive_rew_mean,
            self.episode_num,
        )


    def log_train_metrics_to_tensorboard(self, ep_ret, t, t_start):
        self.tb_logger.add_scalar("train/lr", np.array(self.pi_scheduler.get_last_lr()), self.episode_num)
        self.tb_logger.add_scalar("train/episodic_return", ep_ret, self.episode_num)
        r = [v["episode"]["r"] for v in self.metadata["info"] if v != {}]
        l = [v["episode"]["l"] for v in self.metadata["info"] if v != {}]
        lin_vel_reward = [v["episode"]["lin_vel_reward"] for v in self.metadata["info"] if v != {}]
        collision_penalty = [v["episode"]["collision_penalty"] for v in self.metadata["info"] if v != {}]
        ang_vel_penalty = [v["episode"]["ang_vel_penalty"] for v in self.metadata["info"] if v != {}]
        survive_rew = [v["episode"]["survive_rew"] for v in self.metadata["info"] if v != {}]
        # -----
        r_mean = sum(r) / len(r)
        l_mean = sum(l) / len(l)
        lin_vel_reward_mean = sum(lin_vel_reward) / len(lin_vel_reward)
        collision_penalty_mean = sum(collision_penalty) / len(collision_penalty)
        ang_vel_penalty_mean = sum(ang_vel_penalty) / len(ang_vel_penalty)
        survive_rew_mean = sum(survive_rew) / len(survive_rew)

        self.tb_logger.add_scalar(
            "train/ep_rew_mean",
            r_mean,
            self.episode_num,
        )
        self.tb_logger.add_scalar(
            "train/ep_length",
            l_mean,
            self.episode_num,
        )
        self.tb_logger.add_scalar(
            "train/ep_lin_vel_reward",
            lin_vel_reward_mean,
            self.episode_num,
        )
        self.tb_logger.add_scalar(
            "train/ep_collision_penalty",
            collision_penalty_mean,
            self.episode_num,
        )
        self.tb_logger.add_scalar(
            "train/ep_ang_vel_penalty",
            ang_vel_penalty_mean,
            self.episode_num,
        )
        self.tb_logger.add_scalar(
            "train/ep_survive_rew",
            survive_rew_mean,
            self.episode_num,
        )
       
        self.tb_logger.add_scalar("train/ep_n_steps", t - t_start, self.episode_num)
