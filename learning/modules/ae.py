import torch
import torch.nn as nn
import torch.nn.functional as F


class AE(nn.Module):
    """
    AE
    ----------------
    A deterministic autoencoder compatible with the FLD interface.
    Inputs/outputs keep the same shapes as the previous VAE-based AE,
    but encoding is deterministic (no μ / logσ² or reparameterization).

    Attributes:
        input_channel (int): Number of input channels (observation dimensions).
        history_horizon (int): Length of the input time-series history.
        latent_channel (int): Number of latent channels for encoding.
        device (torch.device): Device to run the model on (e.g., 'cpu' or 'cuda').
        dt (float): Time step between observations.
        encoder_shape (list): Shape of the encoder layers.
        decoder_shape (list): Shape of the decoder layers.
        encoder (nn.Sequential): Encoder network for feature extraction.
        fc_latent (nn.Linear): Linear layer mapping flattened encoder output to latent.
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

        # Flatten then map to deterministic latent vector
        latent_input_dim = curr_in * self.history_horizon
        self.fc_latent = nn.Linear(latent_input_dim, latent_channel).to(self.device)

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
        Encode input deterministically into latent vector.

        Args:
            x (torch.Tensor): Input with shape (B, input_channel, history_horizon)

        Returns:
            torch.Tensor: Latent vector of shape (B, latent_channel)
        """
        h = self.encoder(x)
        h = torch.flatten(h, start_dim=1)
        z = self.fc_latent(h)
        return z

    def decode(self, z):
        """Decode latent vector to reconstructed signal."""
        h = self.fc_decode(z)

        # reshape back to (B, hidden_dim, T)
        h = h.view(z.size(0), -1, self.history_horizon)
        x_recon = self.decoder(h)
        return x_recon

    def forward(self, x, k=1):
        """
        Forward pass of the deterministic AE model.

        Args:
            x: (B, input_channel, history_horizon)
            k: number of prediction steps (for interface consistency)

        Returns:
            predict, latent, signal, params

        Notes:
            - `params` now contains only the latent space tensor (preserving interface).
            - Frequency/amplitude/offset synthesis removed; only the latent space is kept.
        """
        z = self.encode(x)
        x_recon = self.decode(z)

        # Maintain interface compatibility with FLD:
        latent = z
        signal = x_recon  # corresponds to FLD's "signal"

        # For prediction, assume future k steps equal current reconstruction (AE does not predict dynamics)
        predict = x_recon.unsqueeze(0).repeat(k, 1, 1, 1)

        return predict, latent, signal

    def get_dynamics_error(self, state_transitions, k):
        """
        Dynamics error evaluation compatible with the FLD interface.
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
            predict, _, _, _ = self.forward(
                state_transitions_sequence.flatten(0, 1).swapaxes(1, 2), k
            )

        predict = predict.swapaxes(2, 3).view(
            k,
            -1,
            state_transitions.size(1) - self.history_horizon + 1,
            self.history_horizon,
            state_transitions.size(2),
        )
        error = torch.zeros(
            state_transitions.size(0),
            device=self.device,
            dtype=torch.float,
            requires_grad=False,
        )

        for i in range(k):
            error[:] += torch.square(
                (predict[i, :, :state_transitions_sequence.size(1) - i] - state_transitions_sequence[:, i:])
            ).mean(dim=(1, 2, 3))

        return error
