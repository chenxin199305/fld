"""
This module contains the `AETraining` class, which is responsible for training the AE (Fourier Latent Dynamics) model.
It includes methods for training, saving, loading, and evaluating the model, as well as fitting a Gaussian Mixture Model (GMM)
to the latent parameterization of state transitions.

Dependencies:
- PyTorch for tensor operations and model training.
- Matplotlib for plotting.
- TensorBoard for logging training metrics.
- Custom modules: AE, Plotter, GaussianMixture, ReplayBuffer, DistributionBuffer.
"""

from learning.modules.ae import AE
from learning.modules.plotter import Plotter
from learning.modules.gmm import GaussianMixture
from learning.storage.replay_buffer import ReplayBuffer

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"


class AETraining:
    """
    Class for training the AE model.

    Args:
        log_dir (str): The directory to save the training logs.
        latent_dim (int): The dimension of the latent space.
        history_horizon (int): The length of the input observation window.
        forecast_horizon (int): The number of consecutive future steps to predict while maintaining the quasi-constant latent parameterization.
        state_idx_dict (dict): A dictionary mapping state names to their corresponding indices.
        state_transitions_data (torch.Tensor): The state transitions data.
        state_transitions_mean (torch.Tensor): The mean of the state transitions data.
        state_transitions_std (torch.Tensor): The standard deviation of the state transitions data.
        ae_encoder_shape (list, optional): The shape of the AE encoder. Defaults to None.
        ae_decoder_shape (list, optional): The shape of the AE decoder. Defaults to None.
        ae_learning_rate (float, optional): The learning rate for AE optimization. Defaults to 0.0001.
        ae_weight_decay (float, optional): The weight decay for AE optimization. Defaults to 0.0005.
        ae_num_mini_batches (int, optional): The number of mini-batches for AE training. Defaults to 80.
        device (str, optional): The device to use for training. Defaults to "cuda".
        loss_function (str, optional): The loss function to use. Can be "mse" or "geometric". Defaults to "mse".
        noise_level (float, optional): The level of noise to add to the input data. Defaults to 0.0.
        loss_horizon_discount (float, optional): The discount factor for the loss horizon. Defaults to 1.0.
    """

    def __init__(self,
                 log_dir,
                 latent_dim,
                 history_horizon,
                 forecast_horizon,
                 state_idx_dict,
                 state_transitions_data,
                 state_transitions_mean,
                 state_transitions_std,
                 ae_encoder_shape=None,
                 ae_decoder_shape=None,
                 ae_learning_rate=0.0001,
                 ae_weight_decay=0.0005,
                 ae_num_mini_batches=80,
                 device="cuda",
                 loss_function="mse",  # mse or geometric
                 noise_level=0.0,
                 loss_horizon_discount=1.0,
                 ) -> None:
        """
        Initializes the AETraining class and its components, including the AE model, optimizer, buffers, and plotting utilities.
        """

        # num_steps denotes the trajectory length induced by bootstrapping the window of history_horizon forward with forecast_horizon steps
        # num_groups denotes the number of such num_steps
        self.num_motions, \
            self.num_trajs, \
            self.num_groups, \
            self.num_steps, \
            self.observation_dim \
            = state_transitions_data.size()

        print(
            f"[AETraining] \n"
            f"state_transitions_data: \n"
            f"num_motions: {self.num_motions}\n"
            f"num_trajs: {self.num_trajs}\n"
            f"num_groups: {self.num_groups}\n"
            f"num_steps: {self.num_steps}\n"
            f"observation_dim: {self.observation_dim}\n"
        )

        self.log_dir = log_dir
        self.latent_dim = latent_dim
        self.history_horizon = history_horizon
        self.forecast_horizon = forecast_horizon
        self.state_transitions_data = state_transitions_data
        self.state_transitions_mean = state_transitions_mean
        self.state_transitions_std = state_transitions_std
        self.ae_num_mini_batches = ae_num_mini_batches
        self.device = device
        self.loss_function = loss_function
        self.noise_level = noise_level
        self.loss_horizon_discount = loss_horizon_discount
        self.writer = SummaryWriter(log_dir=log_dir, flush_secs=10)

        # Initialize loss scaling factors based on state indices
        self.loss_state_idx_dict = {}
        current_length = 0
        for state, ids in state_idx_dict.items():
            if (state != "base_pos") and (state != "base_quat"):
                self.loss_state_idx_dict[state] = list(range(current_length, current_length + len(ids)))
                current_length = current_length + len(ids)

        self.loss_scale = torch.ones(1, self.history_horizon, self.observation_dim, device=self.device, dtype=torch.float, requires_grad=False)
        if self.loss_function == "geometric":
            for state, ids in self.loss_state_idx_dict.items():
                if "base_lin_vel" in state:
                    self.loss_scale[..., ids] = 2.0
                elif "base_ang_vel" in state:
                    self.loss_scale[..., ids] = 0.5
                elif "projected_gravity" in state:
                    self.loss_scale[..., ids] = 1.0
                elif "dof_pos" in state:
                    self.loss_scale[..., ids] = 1.0
                elif "dof_vel" in state:
                    self.loss_scale[..., ids] = 0.5
        self.loss_scale *= torch.pow(self.loss_horizon_discount, torch.arange(self.history_horizon, device=self.device, dtype=torch.float, requires_grad=False)).view(1, -1, 1)

        # Initialize the AE model and optimizer
        self.ae = AE(self.observation_dim, self.history_horizon, self.latent_dim, self.device, encoder_shape=ae_encoder_shape, decoder_shape=ae_decoder_shape)
        self.ae_optimizer = optim.Adam(self.ae.parameters(), lr=ae_learning_rate, weight_decay=ae_weight_decay)

        # Initialize replay buffers
        self.replay_buffer_size = self.num_motions * self.num_trajs * self.num_groups
        self.state_transitions = ReplayBuffer(self.observation_dim,
                                              self.num_steps,
                                              self.replay_buffer_size,
                                              self.device)
        self.state_transitions.insert(self.state_transitions_data.flatten(0, 2))

        # Initialize plotting utilities
        self.plotter = Plotter()
        self.fig0, self.ax0 = plt.subplots(1, 3)
        self.fig1, self.ax1 = plt.subplots(6, 1)
        self.fig2, self.ax2 = plt.subplots(8, 5)
        self.fig3, self.ax3 = plt.subplots()

        self.current_learning_iteration = 0

    def compute_loss(self, input, target):
        """
        Compute the loss between the input and target tensors.

        Args:
            input (torch.Tensor): The input tensor.
            target (torch.Tensor): The target tensor.

        Returns:
            torch.Tensor: The computed loss.
        """
        input_original = input * self.state_transitions_std + self.state_transitions_mean
        target_original = target * self.state_transitions_std + self.state_transitions_mean
        return torch.mean(torch.sum(torch.square((input_original - target_original) * self.loss_scale), dim=-1))

    def train(self, max_iterations=1000):
        """
        Train the AE model.

        Args:
            max_iterations (int, optional): The maximum number of training iterations. Defaults to 1000.
        """
        print(
            f"{GREEN}{BOLD}"
            f"[AETraining] "
            f"Training started."
            f"{RESET}"
        )

        # reset the iterations information
        tot_iter = self.current_learning_iteration + max_iterations

        # reset loss
        mean_ae_loss = 0

        for it in range(self.current_learning_iteration, tot_iter):

            print(
                f"[AETraining] \n"
                f"it = {it}\n"
                f"max_iterations = {max_iterations}\n"
                f"current_learning_iteration = {self.current_learning_iteration}\n"
                f"num_motions = {self.num_motions}\n"
                f"num_trajs = {self.num_trajs}\n"
                f"num_groups = {self.num_groups}\n"
                f"num_mini_batches = {self.ae_num_mini_batches}\n"
            )

            """
            Jason 2025-11-10:
            把 self.num_motions * self.num_trajs * self.num_groups 分成 ae_num_mini_batches 份，
            每一份的大小是 self.num_motions * self.num_trajs * self.num_groups // self.ae_num_mini_batches
            这样做的目的是为了在每次迭代中，能够更好地利用数据进行训练，同时也能控制每个 mini-batch 的大小，避免内存溢出。
            这种划分方式确保了每个 mini-batch 都包含足够多的数据样本，从而提高训练的稳定性和效果。
            另外，这种划分方式也有助于模型更好地捕捉数据的多样性，因为每个 mini-batch 都来自于整个数据集。
            综上所述，这种划分方式在实践中被证明是有效的，能够提升模型的训练效果和泛化能力。
            """
            state_transitions_data_generator = self.state_transitions.feed_forward_generator(
                self.ae_num_mini_batches,
                self.num_motions * self.num_trajs * self.num_groups // self.ae_num_mini_batches
            )

            # --------------------------------------------------

            for batch_state_transitions in state_transitions_data_generator:

                # Do number of mini_batches updates per iteration
                # batch_state_transitions : (mini_batch_size, num_steps, obs_dim)

                # batch: (mini_batch_size, forecast_horizon, obs_dim, history_horizon)
                batch = batch_state_transitions.unfold(dimension=1,
                                                       size=self.history_horizon,
                                                       step=1)

                # add noise to the input data
                batch_noised = batch + torch.randn_like(batch, device=self.device) * self.noise_level

                # (mini_batch_size, obs_dim, history_horizon)
                batch_input = batch_noised[:, 0, :, :]

                print(
                    f"[AETraining] \n"
                    f"batch_state_transitions.shape: {batch_state_transitions.shape}\n"
                    f"batch.shape: {batch.shape}\n"
                    f"batch_noised.shape: {batch_noised.shape}\n"
                    f"batch_input.shape: {batch_input.shape}\n"
                )

                # predict: (forecast_horizon, mini_batch_size, obs_dim, history_horizon)
                # latent: (mini_batch_size, latent_dim, history_horizon)
                # signal: (mini_batch_size, latent_dim, history_horizon)
                predict, \
                    latent, \
                    signal, \
                    = self.ae.forward(batch_input, k=self.forecast_horizon)

                # reconstruction loss
                loss = 0
                for i in range(self.forecast_horizon):
                    # compute loss for each step of forecast_horizon
                    reconstruction_loss = self.compute_loss(
                        # .swapaxes(-2, -1) 把 (obs_dim, history_horizon)
                        # 换成 (history_horizon, obs_dim)，使得时间维在前。
                        predict[i, :, :, :].swapaxes(-2, -1),
                        batch.swapaxes(-2, -1)[:, i],
                    )
                    loss += reconstruction_loss

                mean_ae_loss += loss.item()

                # Backpropagation and optimization step
                self.ae_optimizer.zero_grad()
                loss.backward()
                self.ae_optimizer.step()

            ae_num_updates = self.ae_num_mini_batches
            mean_ae_loss /= ae_num_updates

            # --------------------------------------------------

            self.writer.add_scalar(f"ae/loss", mean_ae_loss, it)

            print(f"[AETraining] Training iteration {it}/{self.current_learning_iteration + max_iterations}.")

            if it % 50 == 0:
                self.save(it)

                with torch.no_grad():
                    self.ae.eval()

                    plot_traj_index = 0

                    for i in range(self.num_motions):
                        eval_traj = self.state_transitions_data[i, 0, :, :self.history_horizon, :].swapaxes(1, 2)
                        predict, latent, signal = self.ae(eval_traj)
                        predict_current = predict[0]

                        print(
                            f"[AETraining] \n"
                            f"eval_traj.shape = {eval_traj.shape}\n"
                            f"predict.shape = {predict.shape}\n"
                            f"latent.shape = {latent.shape}\n"
                            f"signal.shape = {signal.shape}\n"
                            f"predict_current.shape = {predict_current.shape}\n"
                        )

                        self.plotter.plot_curves(
                            self.ax1[0], eval_traj[plot_traj_index],
                            xmin=-1.0, xmax=1.0, ymin=-5.0, ymax=5.0,
                            title="Motion Curves" + " " + str(self.ae.input_channel) + "x" + str(self.history_horizon),
                            show_axes=False)
                        self.plotter.plot_curves(
                            self.ax1[1], latent[plot_traj_index],
                            xmin=-1.0, xmax=1.0, ymin=-2.0, ymax=2.0,
                            title="Latent Convolutional Embedding" + " " + str(self.latent_dim) + "x" + str(self.history_horizon),
                            show_axes=False)
                        self.plotter.plot_curves(
                            self.ax1[3], signal[plot_traj_index],
                            xmin=-1.0, xmax=1.0, ymin=-2.0, ymax=2.0,
                            title="Latent Parametrized Signal" + " " + str(self.latent_dim) + "x" + str(self.history_horizon),
                            show_axes=False)
                        self.plotter.plot_curves(
                            self.ax1[4], predict_current[plot_traj_index],
                            xmin=-1.0, xmax=1.0, ymin=-5.0, ymax=5.0,
                            title="Curve Reconstruction" + " " + str(self.ae.input_channel) + "x" + str(self.history_horizon),
                            show_axes=False)
                        self.plotter.plot_curves(
                            self.ax1[5], torch.vstack((eval_traj[plot_traj_index].flatten(0, 1), predict_current[plot_traj_index].flatten(0, 1))),
                            xmin=-1.0, xmax=1.0, ymin=-5.0, ymax=5.0,
                            title="Curve Reconstruction (Flattened)" + " " + str(1) + "x" + str(self.ae.input_channel * self.history_horizon),
                            show_axes=False)
                        self.writer.add_figure(
                            f"ae/reconstruction/motion_{i}", self.fig1, it)

                    self.ae.train()

        # --------------------------------------------------

        self.current_learning_iteration += max_iterations
        self.save(self.current_learning_iteration)

        print(
            f"{GREEN}{BOLD}"
            f"[AETraining] "
            f"Training finished."
            f"{RESET}"
        )

    def save(self, it):
        """
        Save the model and training statistics to disk.

        Args:
            it (int): Current training iteration.
        """
        torch.save(
            {
                "state_transitions_mean": self.state_transitions_mean,
                "state_transitions_std": self.state_transitions_std,
            },
            self.log_dir + f"/statistics.pt"
        )
        torch.save(
            {
                "ae_state_dict": self.ae.state_dict(),
                "ae_optimizer_state_dict": self.ae_optimizer.state_dict(),
                "iter": it,
            },
            self.log_dir + f"/model_{it}.pt"
        )

    def load(self, path, load_optimizer=True):
        """
        Load the model and optimizer state from a checkpoint.

        Args:
            path (str): Path to the checkpoint file.
            load_optimizer (bool, optional): Whether to load the optimizer state. Defaults to True.
        """
        print(f"[AETraining] Loading model from: {path}.")

        loaded_dict = torch.load(path)
        self.ae.load_state_dict(loaded_dict["ae_state_dict"])
        if load_optimizer:
            self.ae_optimizer.load_state_dict(loaded_dict["ae_optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]

    def fit_gmm(self, covariance_type="diag"):
        """
        Fit a Gaussian Mixture Model (GMM) to the latent parameterization of state transitions.

        Args:
            covariance_type (str, optional): Covariance type for the GMM. Defaults to "diag".
        """
        # Fit GMM to the latent parameterization of all state transitions
        self.fig4, self.ax4 = plt.subplots(1, 3, subplot_kw=dict(projection='polar'))
        self.gmm = GaussianMixture(self.num_motions, self.latent_dim * 3, device=self.device, covariance_type=covariance_type)
        all_state_transitions = self.state_transitions_data[:, :, :, :self.history_horizon, :].flatten(0, 2).swapaxes(1, 2)  # (num_motions * num_trajs * num_groups, obs_dim, history_horizon)
        with torch.no_grad():
            self.ae.eval()
            _, _, _, all_params = self.ae(all_state_transitions)
        all_frequency = all_params[1]  # (num_motions * num_trajs * num_groups, latent_dim)
        all_amplitude = all_params[2]  # (num_motions * num_trajs * num_groups, latent_dim)
        all_offset = all_params[3]  # (num_motions * num_trajs * num_groups, latent_dim)

        print("[AETraining] GMM fitting started.")

        self.gmm.fit(torch.cat((all_frequency, all_amplitude, all_offset), dim=1))

        print("[AETraining] GMM fitting finished.")

        mu, var = self.gmm.get_block_parameters(self.latent_dim)
        self.plotter.plot_gmm(self.ax4[0], all_frequency.view(self.num_motions, -1, self.latent_dim), mu[0], var[0], title="Frequency GMM")
        self.plotter.plot_gmm(self.ax4[1], all_amplitude.view(self.num_motions, -1, self.latent_dim), mu[1], var[1], title="Amplitude GMM")
        self.plotter.plot_gmm(self.ax4[2], all_offset.view(self.num_motions, -1, self.latent_dim), mu[2], var[2], title="Offset GMM")
        torch.save(
            {
                "gmm_state_dict": self.gmm.state_dict(),
            },
            self.log_dir + f"/gmm.pt"
        )
        plt.show()
