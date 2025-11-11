import torch
import torch.nn as nn


class FLD(nn.Module):
    """
    FLD (Fourier Latent Dynamics) is a PyTorch module designed to encode, decode, and predict dynamics
    in time-series data using Fourier-based latent representations.

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
            device: str,
            dt=0.02,
            encoder_shape=None,
            decoder_shape=None,
            **kwargs,
    ):
        """
        Initializes the FLD model.

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
            print("FLD.__init__ got unexpected arguments, which will be ignored: "
                  + str([key for key in kwargs.keys()]))

        super(FLD, self).__init__()

        self.input_channel = observation_dim
        self.history_horizon = history_horizon
        self.latent_channel = latent_channel
        self.device = device
        self.dt = dt

        """
        Jason 2025-11-10:
        构造一个 长度为 history_horizon 的向量，从：
        - 负半长度时间窗口：-(history_horizon - 1) * dt / 2 到
        - 正半长度时间窗口：(history_horizon - 1) * dt / 2 等间隔。
        这种通常用于：
        - 构建时间轴（如 [-0.5s, ..., 0.5s]）
        - 计算历史窗口的时间偏移
        - 构建 kernel、filter、GAE、motion tracking 里的对称时间窗口
        """
        self.args = torch.linspace(-(history_horizon - 1) * self.dt / 2, (history_horizon - 1) * self.dt / 2, self.history_horizon, dtype=torch.float, device=self.device)

        self.freqs = torch.fft.rfftfreq(history_horizon, device=self.device)[1:] * history_horizon
        self.encoder_shape = encoder_shape if encoder_shape is not None else [int(self.input_channel / 3)]
        self.decoder_shape = decoder_shape if decoder_shape is not None else [int(self.input_channel / 3)]

        # --- Encoder ---
        encoder_layers = []
        curr_in_channel = self.input_channel
        for hidden_channel in self.encoder_shape:
            encoder_layers.append(
                nn.Conv1d(
                    curr_in_channel,
                    hidden_channel,
                    history_horizon,
                    stride=1,
                    padding=int((history_horizon - 1) / 2),
                    dilation=1,
                    groups=1,
                    bias=True,
                    padding_mode='zeros')
            )
            encoder_layers.append(nn.BatchNorm1d(num_features=hidden_channel))
            encoder_layers.append(nn.ELU())
            curr_in_channel = hidden_channel
        encoder_layers.append(
            nn.Conv1d(
                self.encoder_shape[-1],
                latent_channel,
                history_horizon,
                stride=1,
                padding=int((history_horizon - 1) / 2),
                dilation=1,
                groups=1,
                bias=True,
                padding_mode='zeros')
        )
        encoder_layers.append(nn.BatchNorm1d(num_features=latent_channel))
        encoder_layers.append(nn.ELU())

        self.encoder = nn.Sequential(*encoder_layers).to(self.device)
        self.encoder.train()

        self.phase_encoder = nn.ModuleList()
        for _ in range(latent_channel):
            phase_encoder_layers = []
            phase_encoder_layers.append(nn.Linear(history_horizon, 2))
            phase_encoder_layers.append(nn.BatchNorm1d(num_features=2))
            phase_encoder = nn.Sequential(*phase_encoder_layers).to(self.device)
            self.phase_encoder.append(phase_encoder)
        self.phase_encoder.train()

        # --- Decoder ---
        decoder_layers = []
        curr_in_channel = latent_channel
        for hidden_channel in self.decoder_shape:
            decoder_layers.append(
                nn.Conv1d(
                    curr_in_channel,
                    hidden_channel,
                    history_horizon,
                    stride=1,
                    padding=int((history_horizon - 1) / 2),
                    dilation=1,
                    groups=1,
                    bias=True,
                    padding_mode='zeros')
            )
            decoder_layers.append(nn.BatchNorm1d(num_features=hidden_channel))
            decoder_layers.append(nn.ELU())
            curr_in_channel = hidden_channel
        decoder_layers.append(
            nn.Conv1d(
                self.decoder_shape[-1],
                self.input_channel,
                history_horizon,
                stride=1,
                padding=int((history_horizon - 1) / 2),
                dilation=1,
                groups=1,
                bias=True,
                padding_mode='zeros')
        )
        self.decoder = nn.Sequential(*decoder_layers).to(self.device)
        self.decoder.train()

    def forward(self, x, k=1):
        """
        Forward pass of the FLD model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channel, history_horizon).
            k (int, optional): Number of prediction steps. Defaults to 1.

        Returns:
            tuple: Predicted dynamics, latent representation, reconstructed signal, and parameters.
            [
                pred_dynamics (torch.Tensor): 未来 k 步的重建结果 (k, batch_size, input_channel, history_horizon).
                latent (torch.Tensor): 输入 x 的 latent 表示 (batch_size, latent_channel, history_horizon).
                signal (torch.Tensor): t=0 时刻的重建信号 (batch_size, latent_channel, history_horizon).
                params (list): 包含相位、频率、振幅、偏移的列表 (phase, frequency, amplitude, offset)，
                                 每个元素形状均为 (batch_size, latent_channel)。
            ]
        """

        # (batch, input_channel, history_horizon)
        # -> (batch, latent_channel, history_horizon)
        x = self.encoder(x)
        latent = x

        # --------------------------------------------------
        # Fourier Transform to get frequency, amplitude, offset
        frequency, amplitude, offset = self.fft(x)
        phase = torch.zeros((x.size(0),
                             self.latent_channel),
                            device=self.device,
                            dtype=torch.float)

        for i in range(self.latent_channel):
            phase_shift = self.phase_encoder[i](x[:, i, :])
            phase[:, i] = torch.atan2(phase_shift[:, 1], phase_shift[:, 0]) / (2 * torch.pi)

        # (batch_size, latent_channel)
        params = [phase, frequency, amplitude, offset]

        # 构建未来 k 步的相位演化, 线性相位增长（harmonic oscillator 方程）
        # φ(t + Δt) = φ(t) + f * dt * Δt
        # (k, batch_size, latent_channel)
        phase_dynamics = phase.unsqueeze(0) \
                         + frequency.unsqueeze(0) * self.dt * torch.arange(0,
                                                                           k,
                                                                           device=self.device,
                                                                           dtype=torch.float,
                                                                           requires_grad=False
                                                                           ).view(-1, 1, 1)

        # 计算未来 k 步完整的时域 latent 波形 z
        # (k, batch_size, latent_channel, history_horizon)
        z = amplitude.unsqueeze(-1).unsqueeze(0) * torch.sin(
            2 * torch.pi * (
                    (frequency.unsqueeze(-1) * self.args).unsqueeze(0)
                    + phase_dynamics.unsqueeze(-1))) \
            + offset.unsqueeze(-1).unsqueeze(0)

        # t=0 时刻的重建, 这是 latent 的重建（类似 autoencoder 的 recon）。
        signal = z[0]
        # --------------------------------------------------

        # decoder 把 latent 动态变回原始空间
        # (k, batch_size, input_channel, history_horizon)
        pred_dynamics = self.decoder(z.flatten(0, 1)).view(k, -1, self.input_channel, self.history_horizon)

        return pred_dynamics, latent, signal, params

    def fft(self, x):
        """
        Computes the Fourier Transform of the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, latent_channel, history_horizon).

        Returns:
            tuple: Frequency, amplitude, and offset of the Fourier Transform.
        """
        rfft = torch.fft.rfft(x, dim=2)
        magnitude = rfft.abs()
        spectrum = magnitude[:, :, 1:]
        power = torch.square(spectrum)
        frequency = torch.sum(self.freqs * power, dim=2) / torch.sum(power, dim=2)
        amplitude = 2 * torch.sqrt(torch.sum(power, dim=2)) / self.history_horizon
        offset = rfft.real[:, :, 0] / self.history_horizon
        return frequency, amplitude, offset

    def get_dynamics_error(self, state_transitions, k):
        """
        Computes the dynamics prediction error.

        Args:
            state_transitions (torch.Tensor): Input state transitions of shape (batch_size, sequence_length, input_channel).
            k (int): Number of prediction steps.

        Returns:
            torch.Tensor: Dynamics prediction error for each batch.
        """
        self.eval()

        state_transitions_sequence = torch.zeros(
            state_transitions.size(0),
            state_transitions.size(1) - self.history_horizon + 1,
            self.history_horizon,
            state_transitions.size(2),
            dtype=torch.float,
            device=self.device,
            requires_grad=False
        )

        for step in range(state_transitions.size(1) - self.history_horizon + 1):
            state_transitions_sequence[:, step] = state_transitions[:, step:step + self.history_horizon, :]

        with torch.no_grad():
            pred_dynamics, _, _, _ = self.forward(state_transitions_sequence.flatten(0, 1).swapaxes(1, 2), k)

        pred_dynamics = pred_dynamics.swapaxes(2, 3).view(k, -1, state_transitions.size(1) - self.history_horizon + 1, self.history_horizon, state_transitions.size(2))
        error = torch.zeros(state_transitions.size(0), device=self.device, dtype=torch.float, requires_grad=False)

        for i in range(k):
            error[:] += torch.square((pred_dynamics[i, :, :state_transitions_sequence.size(1) - i] - state_transitions_sequence[:, i:])).mean(dim=(1, 2, 3))

        return error
