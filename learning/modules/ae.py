import torch
import torch.nn as nn
import torch.nn.functional as F


class AE(nn.Module):
    """
    AE
    ----------------
    一个与 FLD 类兼容的自编码器模型。
    输入输出接口保持一致，但内部使用 AE 编码和采样机制，而非傅里叶分解。

    Attributes:
        input_channel (int): Number of input channels (observation dimensions).
        history_horizon (int): Length of the input time-series history.
        latent_channel (int): Number of latent channels for encoding.
        device (torch.device): Device to run the model on (e.g., 'cpu' or 'cuda').
        dt (float): Time step between observations.
        args (torch.Tensor): Time arguments for Fourier transformations.
        freqs (torch.Tensor): Frequencies for Fourier transformations.
        encoder_shape (list): Shape of the encoder layers.
        decoder_shape (list): Shape of the decoder layers.
        encoder (nn.Sequential): Encoder network for feature extraction.
        phase_encoder (nn.ModuleList): Phase encoder for latent dynamics.
        decoder (nn.Sequential): Decoder network for reconstructing input signals.
    """

    def __init__(
            self,
            observation_dim,
            history_horizon,
            latent_channel,
            device,
            dt=0.02,
            encoder_shape=None,
            decoder_shape=None,
            **kwargs,
    ):
        """
        Initializes the AE model.

        Args:
            observation_dim (int): Number of input channels (observation dimensions).
            history_horizon (int): Length of the input time-series history.
            latent_channel (int): Number of latent channels for encoding.
            device (torch.device): Device to run the model on (e.g., 'cpu' or 'cuda').
            dt (float, optional): Time step between observations. Defaults to 0.02.
            encoder_shape (list, optional): Shape of the encoder layers. Defaults to None.
            decoder_shape (list, optional): Shape of the decoder layers. Defaults to None.
            **kwargs: Additional arguments (ignored).
        """
        if kwargs:
            print("AE.__init__ got unexpected arguments (ignored): "
                  + str([key for key in kwargs.keys()]))

        super(AE, self).__init__()

        self.input_channel = observation_dim
        self.history_horizon = history_horizon
        self.latent_channel = latent_channel
        self.device = device
        self.dt = dt

        self.encoder_shape = encoder_shape if encoder_shape is not None else [int(self.input_channel / 2)]
        self.decoder_shape = decoder_shape if decoder_shape is not None else [int(self.input_channel / 2)]

        # --- Encoder ---
        encoder_layers = []
        curr_in = self.input_channel
        for hidden in self.encoder_shape:
            encoder_layers.append(nn.Conv1d(curr_in, hidden, 3, stride=1, padding=1))
            encoder_layers.append(nn.BatchNorm1d(hidden))
            encoder_layers.append(nn.ELU())
            curr_in = hidden
        self.encoder = nn.Sequential(*encoder_layers).to(self.device)

        # Flatten then map to latent μ, logσ²
        latent_input_dim = curr_in * self.history_horizon
        self.fc_mu = nn.Linear(latent_input_dim, latent_channel).to(self.device)
        self.fc_logvar = nn.Linear(latent_input_dim, latent_channel).to(self.device)

        # --- Decoder ---
        decoder_input_dim = latent_channel
        self.fc_decode = nn.Linear(decoder_input_dim, curr_in * self.history_horizon).to(self.device)

        decoder_layers = []
        curr_in = curr_in
        for hidden in self.decoder_shape:
            decoder_layers.append(nn.Conv1d(curr_in, hidden, 3, stride=1, padding=1))
            decoder_layers.append(nn.BatchNorm1d(hidden))
            decoder_layers.append(nn.ELU())
            curr_in = hidden
        decoder_layers.append(nn.Conv1d(curr_in, self.input_channel, 3, stride=1, padding=1))
        self.decoder = nn.Sequential(*decoder_layers).to(self.device)

    def encode(self, x):
        """
        Encode input into latent mean and logvar.
        """
        h = self.encoder(x)
        h = torch.flatten(h, start_dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Sample z via reparameterization trick.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decode latent vector to reconstructed signal."""
        h = self.fc_decode(z)

        # reshape back to (B, hidden_dim, T)
        h = h.view(z.size(0), -1, self.history_horizon)
        x_recon = self.decoder(h)
        return x_recon

    def forward(self, x, k=1):
        """
        Forward pass of the AE model.

        Args:
            x: (B, input_channel, history_horizon)
            k: number of prediction steps (for interface consistency)

        Returns:
            pred_dynamics, latent, signal, params
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)

        # 保持接口一致
        latent = z
        signal = x_recon  # 对应 FLD 的“信号”

        # ====== 🔧 新增部分：将 AE 参数扩展成 FLD 格式 ======
        # 这里我们没有真正的 phase / frequency / amplitude / offset，
        # 所以造出四个形状匹配的张量以保持接口一致。
        phase = mu
        frequency = logvar
        amplitude = torch.ones_like(mu, device=mu.device)
        offset = torch.zeros_like(mu, device=mu.device)

        params = [phase, frequency, amplitude, offset]

        # 对于预测部分，这里假设未来 k 步与当前重建相同（AE 不预测时序）
        pred_dynamics = x_recon.unsqueeze(0).repeat(k, 1, 1, 1)

        return pred_dynamics, latent, signal, params

    def get_dynamics_error(self, state_transitions, k):
        """
        与 FLD 接口兼容的动态误差评估。
        """
        self.eval()
        state_transitions_sequence = torch.zeros(
            state_transitions.size(0),
            state_transitions.size(1) - self.history_horizon + 1,
            self.history_horizon,
            state_transitions.size(2),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        for step in range(state_transitions.size(1) - self.history_horizon + 1):
            state_transitions_sequence[:, step] = state_transitions[:, step:step + self.history_horizon, :]
        with torch.no_grad():
            pred_dynamics, _, _, _ = self.forward(
                state_transitions_sequence.flatten(0, 1).swapaxes(1, 2), k
            )
        pred_dynamics = pred_dynamics.swapaxes(2, 3).view(
            k,
            -1,
            state_transitions.size(1) - self.history_horizon + 1,
            self.history_horizon,
            state_transitions.size(2),
        )
        error = torch.zeros(state_transitions.size(0), device=self.device, dtype=torch.float)
        for i in range(k):
            error[:] += torch.square(
                (pred_dynamics[i, :, :state_transitions_sequence.size(1) - i] - state_transitions_sequence[:, i:])
            ).mean(dim=(1, 2, 3))
        return error

    def vae_loss(self, recon_x, x, mu, logvar, beta=1.0):
        """标准 AE 损失"""
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + beta * kl
