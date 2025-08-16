import torch
from torch import nn
from einops import rearrange, repeat


torch.hub._validate_not_a_forked_repo = lambda a, b, c: True


class DINOWM(torch.nn.Module):
    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        proprio_encoder,
        frameskip,
        history_size,
        action_dim,
        action_emb_dim,
        proprio_dim,
        proprio_emb_dim,
        device="cpu",
        # boring ...
        action_mean=0,
        action_std=1,
        proprio_mean=0,
        proprio_std=1,
    ):
        super().__init__()
        self.device = device
        self.backbone = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.proprio_encoder = proprio_encoder

        self.frameskip = frameskip

        self.history_size = history_size

        self.proprio_dim = proprio_dim
        self.original_action_dim = action_dim
        self.action_dim = action_dim * frameskip  #

        self.action_emb_dim = action_emb_dim
        self.proprio_emb_dim = proprio_emb_dim

        self.action_mean = action_mean
        self.action_std = action_std
        self.proprio_mean = proprio_mean
        self.proprio_std = proprio_std

    def forward(self, obs, actions):
        z = self.encode(obs, actions)
        z_pred = self.predict(z[:, -self.history_size :])  # (B, hist_size, P, D)
        z_new = z_pred[:, -1:, :, :]  # (B, 1, P, D)

        z_obs, z_act = self.split_embeddings(z_new)

    def encode(self, obs, actions):
        z_obs = self.encode_obs(obs)
        z_pixels = z_obs["pixels"]  # (B, T, P, D)
        z_proprio = z_obs["proprio"]  # (B, T, P_emb)
        z_act = self.encode_action(actions)  # (B, T, A_emb)

        # -- merge state, action, proprio
        n_patches = z_pixels.shape[2]

        # share action/proprio embedding accros patches for each time step
        propr_tiled = repeat(z_proprio.unsqueeze(2), "b t 1 d -> b t p d", p=n_patches)
        act_tiled = repeat(z_act.unsqueeze(2), "b t 1 d -> b t p d", p=n_patches)

        # (B, T, P, dim+A_emb+P_emb)
        z = torch.cat([z_pixels, propr_tiled, act_tiled], dim=3)

        return z  # (B, T, P, d)

    def encode_action(self, actions):
        return self.action_encoder(actions)

    def encode_proprio(self, proprio):
        return self.proprio_encoder(proprio)

    def encode_obs(self, obs):
        """Preprocess observation from the environment obs dict
        Args:
            states: dict with all the observation modalities (img, proprio, etc)
        Returns:
            z: encoded latent states for observations
        """

        pixels = obs["pixels"].float()  # (B, T, 3, H, W)
        pixels = pixels.unsqueeze(1) if pixels.ndim == 4 else pixels
        B = pixels.shape[0]
        pixels = rearrange(pixels, "b t ... -> (b t) ...")
        proprio = obs["proprio"].float()  # (B, T, P)

        # ? transform pixels ?

        # -- pixels embeddings
        z_pixels = self.backbone(pixels)
        z_pixels = z_pixels[:, 1:, :]  # drop cls token
        z_pixels = rearrange(z_pixels, "(b t) p d -> b t p d", b=B)

        # -- proprio embeddings
        z_proprio = self.encode_proprio(proprio)  # (B, T, P) -> (B, T, P_emb)

        return {"pixels": z_pixels, "proprio": z_proprio}

    def predict(self, z):
        """predict next latent state
        Args:
            z: (B, T, P, d)
        Returns:
            preds: (B, T, P, d)
        """
        T = z.shape[1]
        z = rearrange(z, "b t p d -> b (t p) d")
        preds = self.predictor(z)
        preds = rearrange(preds, "b (t p) d -> b t p d", t=T)
        return preds

    # def forward(self, obs, actions):  # obs, actions, proprio):
    #     """world model forward pass.
    #     Args:
    #         obs: dict with all the observation modalities (img, proprio, etc)
    #     Returns:
    #         z_pred: predicted next latent states (B, T, P, d)
    #     """

    #     obs = torch.from_numpy(states["pixels"]).float()
    #     proprio = torch.from_numpy(states["proprio"]).float()

    #     # normalize proprio and actions
    #     actions = self.normalize_actions(actions)
    #     proprio = self.normalize_proprio(proprio)

    #     # add dummy temporal dimension if needed
    #     if obs.ndim == 4:
    #         obs = obs.unsqueeze(1)  # (B, T, C, H, W)
    #         proprio = proprio.unsqueeze(1)  # (B, T, P)
    #         actions = actions.unsqueeze(1)  # (B, T, A)

    #     # -- move to device
    #     obs = obs.to(self.device)
    #     actions = actions.to(self.device)
    #     proprio = proprio.to(self.device)

    #     # -- embed dim
    #     action_emb_dim = actions.shape[-1]
    #     proprio_emb_dim = proprio.shape[-1]

    #     # -- preprocess inputs
    #     if type(actions) is dict:
    #         actions = [a.flatten(2) for a in actions.values()]
    #         actions = torch.cat(actions, -1)
    #     else:
    #         actions.flatten(2)

    #     # -- infer next state
    #     z = self.encode(obs, actions, proprio)
    #     z_preds = self.predict(z)

    #     # TODO should check from their code
    #     # z_src = z[:, : Config.num_hist, :, :]
    #     # z_tgt = z[:, Config.num_pred :, :, :]

    #     # keep only the part corresponding to the visual features
    #     # TODO chekc if need to remove proprio as well
    #     z_pred_visual = z_preds[..., : -action_emb_dim - proprio_emb_dim]

    #     return z_pred_visual

    def replace_actions_from_z(self, z, act):
        """Replace the action embeddings in the latent state z with the provided actions."""
        n_patches = z.shape[2]
        z_act = self.encode_act(act)
        act_tiled = repeat(z_act.unsqueeze(2), "b t 1 a -> b t p a", p=n_patches)
        # z (B, T, P, d) with d = dim + proprio_emb_dim + action_emb_dim
        # replace the last 'action_emb_dim' dims of z with the action embeddings
        z[..., -self.action_emb_dim :] = act_tiled
        return z

    def split_embeddings(self, z):
        """Unmerge embedding z into separate modalities.
        Args:
            z (B, T, P, d) where d = dim + proprio_emb_dim + action_emb_dim
        Returns:
            z_obs: dict with separate modalities
            z_act: (B, T, P, action_emb_dim)
        """

        A = self.action_emb_dim
        P = self.proprio_emb_dim

        z_pixels = z[..., : -(P + A)]  # (B, T, P, D)
        z_proprio = z[:, :, 0, -(P + A) : -A]  # (B, T, 1, P_emb)
        z_act = z[:, :, 0, -A:]  # (B, T, 1, A_emb)

        z_obs = {"pixels": z_pixels, "proprio": z_proprio}
        return z_obs, z_act

    def rollout(self, obs_0, actions):
        """Rollout the world model given an initial observation and a sequence of actions.

        Params:
        obs_start: n current observations (B, n, C, H, W)
        actions: current and predicted actions (B, n+t, action_dim)

        Returns:
        z_obs: dict with latent observations (B, n+t+1, n_patches, D)
        z: predicted latent states (B, n+t+1, n_patches, D)
        """

        n_obs = len(obs_0["pixels"])

        act_0 = actions[:, :n_obs]
        action = actions[:, n_obs:]

        z = self.encode(obs_0, act_0)
        t = 0

        # simulate action taken in the world model
        n_steps = action.shape[1]
        for t in range(n_steps):
            # predict next state based on the history size
            z_pred = self.predict(z[:, -self.history_size :])  # (B, hist_size, P, D)
            z_new = z_pred[:, -1:, :, :]  # (B, 1, P, D)

            print(f"Step {t + 1}/{n_steps} - z_new shape: {z_new.shape}")

            # update z_new with the new action
            next_action = action[:, t + 1, :]  # (B, action_dim)
            z_new = self.replace_actions_from_z(z_new, next_action)
            z = torch.cat([z, z_new], dim=1)  # (B, n+t, P, D)

        # predict n+t+1 state
        z_pred = self.predict(z[:, -self.history_size :])
        z_new = z_pred[:, -1:, :, :]
        z = torch.cat([z, z_new], dim=1)  # (B, n+t+1, P, D)
        z_obs, _ = self.split_embeddings(z)

        return z_obs, z

    def normalize_actions(self, actions):
        """Normalize actions using the defined normalization parameters."""
        return (actions - self.action_mean) / self.action_std

    def normalize_proprio(self, proprio):
        """Normalize proprioceptive data using the defined normalization parameters."""
        return (proprio - self.proprio_mean) / self.proprio_std

    def denormalize_actions(self, actions):
        """Denormalize actions using the defined normalization parameters."""
        return actions * self.action_std + self.action_mean

    def denormalize_proprio(self, proprio):
        """Denormalize proprioceptive data using the defined normalization parameters."""
        return proprio * self.proprio_std + self.proprio_mean


class DinoV2Encoder(nn.Module):
    def __init__(self, name, feature_key):
        super().__init__()
        self.name = name
        self.base_model = torch.hub.load("facebookresearch/dinov2", name)
        self.feature_key = feature_key
        self.emb_dim = self.base_model.num_features
        if feature_key == "x_norm_patchtokens":
            self.latent_ndim = 2
        elif feature_key == "x_norm_clstoken":
            self.latent_ndim = 1
        else:
            raise ValueError(f"Invalid feature key: {feature_key}")

        self.patch_size = self.base_model.patch_size

    def forward(self, x):
        emb = self.base_model.forward_features(x)[self.feature_key]
        if self.latent_ndim == 1:
            emb = emb.unsqueeze(1)  # dummy patch dim
        return emb


class Embedder(torch.nn.Module):
    def __init__(
        self,
        num_frames=1,
        tubelet_size=1,
        in_chans=8,
        emb_dim=10,
    ):
        super().__init__()

        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.in_chans = in_chans
        self.emb_dim = emb_dim

        self.patch_embed = torch.nn.Conv1d(
            in_chans, emb_dim, kernel_size=tubelet_size, stride=tubelet_size
        )

    def forward(self, x):
        x = x.float()
        x = x.permute(0, 2, 1)  # (B, T, B) -> (B, D, T)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)  # (B, D, T) -> (B, T, D)
        return x


class CausalPredictor(nn.Module):
    def __init__(
        self,
        *,
        num_patches,
        num_frames,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        assert pool in {"cls", "mean"}, (
            "pool type must be either cls (cls token) or mean (mean pooling)"
        )

        self.num_patches = num_patches
        self.num_frames = num_frames

        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_frames * (num_patches), dim)
        )  # dim for the pos encodings
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout, num_patches, num_frames
        )
        self.pool = pool

    def forward(self, x):  # x: (b, window_size * H/patch_size * W/patch_size, 384)
        b, n, _ = x.shape
        x = x + self.pos_embedding[:, :n]
        x = self.dropout(x)
        x = self.transformer(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(
        self, dim, heads=8, dim_head=64, dropout=0.0, num_patches=1, num_frames=1
    ):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

        # self.register_buffer(
        #     "temp_mask", self.generate_mask_matrix(num_patches, num_frames)
        # )

        self.bias = self.generate_mask_matrix(num_patches, num_frames).to("cuda")

    def forward(self, x):
        B, T, C = x.size()
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        dots = dots.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)

    def generate_mask_matrix(self, npatch, nwindow):
        zeros = torch.zeros(npatch, npatch)
        ones = torch.ones(npatch, npatch)
        rows = []
        for i in range(nwindow):
            row = torch.cat([ones] * (i + 1) + [zeros] * (nwindow - i - 1), dim=1)
            rows.append(row)
        mask = torch.cat(rows, dim=0).unsqueeze(0).unsqueeze(0)
        return mask


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        num_patches=1,
        num_frames=1,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    Attention(
                        dim,
                        heads=heads,
                        dim_head=dim_head,
                        dropout=dropout,
                        num_patches=num_patches,
                        num_frames=num_frames,
                    ),
                    FeedForward(dim, mlp_dim, dropout=dropout),
                ])
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)
