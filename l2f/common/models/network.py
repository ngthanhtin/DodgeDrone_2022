import torch
import torch.nn as nn
from l2f.baselines.core import mlp, SquashedGaussianMLPActor


class Qfunction(nn.Module):
    """
    Modified from the core MLPQFunction and MLPActorCritic to include a state encoder
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # pdb.set_trace()
        self.state_encoder = mlp(
            [55] + self.cfg[self.cfg["use_encoder_type"]]["state_hiddens"]
        )
        self.regressor = mlp(
            [
                self.cfg[self.cfg["use_encoder_type"]]["latent_dims"]
                + self.cfg[self.cfg["use_encoder_type"]]["state_hiddens"][-1]
                + 4 # action_dims
            ]
            + self.cfg[self.cfg["use_encoder_type"]]["hiddens"]
            + [1]
        )

    def forward(self, obs_feat, action):
        img_embed = obs_feat[
            ..., : self.cfg[self.cfg["use_encoder_type"]]["latent_dims"]
        ]  # n x latent_dims
        state = obs_feat[
            ..., self.cfg[self.cfg["use_encoder_type"]]["latent_dims"] :
        ]  # n x 55

        st_embed = self.state_encoder(state)  # n x 8
        
        out = self.regressor(torch.cat([img_embed, st_embed, action], dim=-1))  # n x 1
        
        # return out.view(-1)
        return out.squeeze()

class ActorCritic(nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        cfg,
        activation=nn.ReLU,
        latent_dims=None,
        device="cpu"
    ):
        super().__init__()
        self.cfg = cfg
        obs_dim = observation_space.shape[0] if latent_dims is None else latent_dims
        act_dim = action_space.shape[1] # 0
        act_limit = action_space.high[0]
        
        # build policy and value functions
        self.state_encoder = mlp(
            [55] + self.cfg[self.cfg["use_encoder_type"]]["state_hiddens"]
        )
        
        self.policy = SquashedGaussianMLPActor(
            obs_dim,
            act_dim,
            cfg[cfg["use_encoder_type"]]["actor_hiddens"],
            activation,
            act_limit,
        )

        self.q1 = Qfunction(cfg)
        self.q2 = Qfunction(cfg)

        self.device = device
        self.to(device)

    def pi(self, obs_feat, deterministic=False):
        # if obs_feat.ndimension() == 1:
        #    obs_feat = obs_feat.unsqueeze(0)
        
        img_embed = obs_feat[
            ..., : self.cfg[self.cfg["use_encoder_type"]]["latent_dims"]
        ]  # batch, nenv x latent_dims
        state = obs_feat[
            ..., self.cfg[self.cfg["use_encoder_type"]]["latent_dims"] :
        ]  # batch, nenv x 55
        st_embed = self.state_encoder(state)  # n x 8
        feat = torch.cat([img_embed, st_embed], dim=-1) # n x 40

        pi = self.policy(feat, deterministic, True)
        
        return pi

    def act(self, obs_feat, deterministic=False):        
        with torch.no_grad():
            img_embed = obs_feat[
                ..., : self.cfg[self.cfg["use_encoder_type"]]["latent_dims"]
            ]  # n x latent_dims
            state = obs_feat[
                ..., self.cfg[self.cfg["use_encoder_type"]]["latent_dims"] :
            ]  # n x 55
            st_embed = self.state_encoder(state)  # n x 8
            feat = torch.cat([img_embed, st_embed], dim=-1)
            a, _ = self.policy(feat, deterministic, False)
            
            # a = a.squeeze(0)
        return a.numpy() if self.device == "cpu" else a.cpu().numpy()
