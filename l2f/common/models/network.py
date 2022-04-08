import torch
import torch.nn as nn
from l2f.baselines.core import mlp, SquashedGaussianMLPActor


def resnet18(pretrained=True):
    model = torch.hub.load("pytorch/vision:v0.6.0", "resnet18", pretrained=pretrained)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Identity()
    return model


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
        # self.lr = cfg['resnet']['LR']

    def forward(self, obs_feat, action):
        # if obs_feat.ndimension() == 1:
        #    obs_feat = obs_feat.unsqueeze(0)
        img_embed = obs_feat[
            ..., : self.cfg[self.cfg["use_encoder_type"]]["latent_dims"]
        ]  # n x latent_dims
        state = obs_feat[
            ..., self.cfg[self.cfg["use_encoder_type"]]["latent_dims"] :
        ]  # n x 55

        st_embed = self.state_encoder(state)  # n x 16
        out = self.regressor(torch.cat([img_embed, st_embed, action], dim=-1))  # n x 1
        # pdb.set_trace()
        
        # return out.view(-1)
        return out.squeeze()


class DuelingNetwork(nn.Module):
    """
    Further modify from Qfunction to
        - Add an action_encoder
        - Separate state-dependent value and advantage
            Q(s, a) = V(s) + A(s, a)
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.state_encoder = mlp(
            [55] + self.cfg[self.cfg["use_encoder_type"]]["state_hiddens"]
        )
        self.action_encoder = mlp(
            [4] + self.cfg[self.cfg["use_encoder_type"]]["action_hiddens"]
        )

        n_obs = (
            self.cfg[self.cfg["use_encoder_type"]]["latent_dims"]
            + self.cfg[self.cfg["use_encoder_type"]]["state_hiddens"][-1]
        )
        # self.V_network = mlp([n_obs] + self.cfg[self.cfg['use_encoder_type']]['hiddens'] + [55])
        self.A_network = mlp(
            [n_obs + self.cfg[self.cfg["use_encoder_type"]]["action_hiddens"][-1]]
            + self.cfg[self.cfg["use_encoder_type"]]["hiddens"]
            + [55]
        )
        # self.lr = cfg['resnet']['LR']

    def forward(self, obs_feat, action, advantage_only=False):
        # if obs_feat.ndimension() == 1:
        #    obs_feat = obs_feat.unsqueeze(0)
        img_embed = obs_feat[
            ..., : self.cfg[self.cfg["use_encoder_type"]]["latent_dims"]
        ]  # n x latent_dims
        state = obs_feat[
            ..., self.cfg[self.cfg["use_encoder_type"]]["latent_dims"] :
        ]  # n x 1
        st_embed = self.state_encoder(state)  # n x 16
        action_embed = self.action_encoder(action)

        out = self.A_network(torch.cat([img_embed, st_embed, action_embed], dim=-1))
        """
        if advantage_only == False:
            V = self.V_network(torch.cat([img_embed, st_embed], dim = -1)) # n x 1
            out += V
        """
        return out.view(-1)


class ActorCritic(nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        cfg,
        activation=nn.ReLU,
        latent_dims=None,
        device="cpu",
        safety=False,  ## Flag to indicate architecture for Safety_actor_critic
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
        if safety:
            self.q1 = DuelingNetwork(cfg)
        else:
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
        # feat = feat.view(feat.shape[0], -1)
        # print(feat.shape)
        pi = self.policy(feat, deterministic, True)
        
        return pi

    def act(self, obs_feat, deterministic=False):
        # if obs_feat.ndimension() == 1:
        #    obs_feat = obs_feat.unsqueeze(0)
        
        with torch.no_grad():
            img_embed = obs_feat[
                ..., : self.cfg[self.cfg["use_encoder_type"]]["latent_dims"]
            ]  # n x latent_dims
            state = obs_feat[
                ..., self.cfg[self.cfg["use_encoder_type"]]["latent_dims"] :
            ]  # n x 55
            # pdb.set_trace()
            st_embed = self.state_encoder(state)  # n x 8
            feat = torch.cat([img_embed, st_embed], dim=-1)
            a, _ = self.policy(feat, deterministic, False)
            
            # a = a.squeeze(0)
        return a.numpy() if self.device == "cpu" else a.cpu().numpy()
