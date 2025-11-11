from humanoid_gym import LEGGED_GYM_ROOT_DIR
from scripts.vae.training import VAETraining
import os
import torch

RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"


class VAEExperiment:
    """
    Represents an experiment for VAE (Variational Autoencoder) training on motion data.

    Args:
        state_idx_dict (dict): A dictionary mapping state names to their corresponding indices.
        history_horizon (int): The length of the input observation window.
        forecast_horizon (int): The number of consecutive future steps to predict while maintaining the quasi-constant latent parameterization.
        device (str): The device to use for computation.

    """

    def __init__(self, state_idx_dict, history_horizon, forecast_horizon, device):
        self.state_idx_dict = state_idx_dict
        self.history_horizon = history_horizon
        self.forecast_horizon = forecast_horizon
        self.dim_of_interest = torch.cat(
            [
                torch.tensor(ids, device=device, dtype=torch.long, requires_grad=False)
                for state, ids in state_idx_dict.items()
                if ((state != "base_pos") and (state != "base_quat"))
            ]
        )
        self.device = device

    def prepare_data(self):
        """
        Loads and prepares the motion data.

        This method loads the motion data from the specified directory, normalizes it,
        and calculates the mean and standard deviation of the state transitions data.

        """
        datasets_root = os.path.join(LEGGED_GYM_ROOT_DIR
                                     + "/resources/robots/mit_humanoid/datasets/misc")
        motion_data = os.listdir(datasets_root)
        motion_name_set = [
            data.replace('motion_data_', '').replace('.pt', '')
            for data in motion_data if "combined" not in data and ".pt" in data
        ]

        # --------------------------------------------------

        motion_data_collection = []

        for i, motion_name in enumerate(motion_name_set):
            motion_path = os.path.join(datasets_root, "motion_data_" + motion_name + ".pt")

            # (num_trajs, traj_len, obs_dim)
            motion_data = torch.load(motion_path, map_location=self.device)[:, :, self.dim_of_interest]
            loaded_num_trajs, loaded_num_steps, loaded_obs_dim = motion_data.size()

            print(
                f"[VAEExperiment] "
                f"Loaded motion {i}: "
                f"name {motion_name}, "
                f"with {loaded_num_trajs} trajectories, "
                f"with {loaded_num_steps} steps, "
                f"with {loaded_obs_dim} dimensions."
            )

            motion_data_collection.append(motion_data.unsqueeze(0))

        # (num_motions, num_trajs, traj_len, obs_dim)
        motion_data_collection = torch.cat(motion_data_collection, dim=0)

        print(
            f"{YELLOW}{BOLD}"
            f"[VAEExperiment] "
            f"motion_data_collection.shape: {motion_data_collection.shape}"
            f"{RESET}"
        )

        self.state_transitions_mean = motion_data_collection.flatten(0, 2).mean(dim=0)
        self.state_transitions_std = motion_data_collection.flatten(0, 2).std(dim=0) + 1e-6

        print(
            f"{YELLOW}{BOLD}"
            f"[VAEExperiment] "
            f"state_transitions_mean.shape: {self.state_transitions_mean.shape}"
            f"\n"
            f"[VAEExperiment] "
            f"state_transitions_std.shape: {self.state_transitions_std.shape}"
            f"{RESET}"
        )

        # --------------------------------------------------

        """
        Jason 2025-11-11:
        unfold 的作用是创建一个滑动窗口。

        这里的 dimension=2 表示在第2个维度上进行操作，也就是 traj_len 维度。
        size=self.history_horizon + self.forecast_horizon - 1 表示窗口的大小。
        step=1 表示窗口每次移动1个位置。

        经过 unfold 操作后，原本的 traj_len 维度会被拆分成两个维度：
        一个是 num_groups，表示有多少个这样的窗口；
        另一个是 num_steps，表示每个窗口内的时间步数。
        最后通过 swapaxes(-2, -1) 将这两个新维度的位置交换，使得最终的维度顺序变为
        (num_motions, num_trajs, num_groups, num_steps, obs_dim)。
        这样处理后的数据可以更方便地用于后续的模型训练，特别是对于时间序列数据的处理。

        具体来说，假设 history_horizon=51，forecast_horizon=50，那么
        每个窗口的大小就是 51 + 50 - 1 = 100。
        这样每个窗口包含了51帧的历史数据和50帧的未来数据（减去1是因为历史的最后一帧和未来的第一帧是重叠的）。
        通过这种方式，可以生成多个这样的窗口，用于训练模型预测未来的状态。
        """
        # (num_motions, num_trajs, traj_len, obs_dim)
        # -> unfold -> (num_motions, num_trajs, num_groups, obs_dim, num_steps)
        # -> swapaxes -> (num_motions, num_trajs, num_groups, num_steps, obs_dim)
        motion_data_collection = motion_data_collection.unfold(
            dimension=2,
            size=self.history_horizon + self.forecast_horizon - 1,
            step=1)

        print(
            f"{YELLOW}{BOLD}"
            f"[VAEExperiment] "
            f"After unfold, motion_data_collection.shape: {motion_data_collection.shape}"
            f"{RESET}"
        )

        motion_data_collection = motion_data_collection.swapaxes(-2, -1)

        print(
            f"{YELLOW}{BOLD}"
            f"[VAEExperiment] "
            f"After swapaxes, motion_data_collection.shape: {motion_data_collection.shape}"
            f"{RESET}"
        )

        # (num_motions, num_trajs, num_groups, num_steps, obs_dim)
        self.state_transitions_data = (motion_data_collection - self.state_transitions_mean) / self.state_transitions_std

        print(
            f"{YELLOW}{BOLD}"
            f"[FLDExperiment] "
            f"self.state_transitions_data.shape: {self.state_transitions_data.shape}"
            f"{RESET}"
        )

        # --------------------------------------------------

    def train(self, log_dir, latent_dim):
        """
        Trains the VAE model.

        Args:
            log_dir (str): The directory to save the training logs.
            latent_dim (int): The dimensionality of the latent space.

        """
        vae_training = VAETraining(
            log_dir,
            latent_dim,
            self.history_horizon,
            self.forecast_horizon,
            self.state_idx_dict,
            self.state_transitions_data,
            self.state_transitions_mean,
            self.state_transitions_std,
            vae_encoder_shape=[64, 64],
            vae_decoder_shape=[64, 64],
            vae_learning_rate=0.0001,
            vae_weight_decay=0.0005,
            vae_num_mini_batches=10,
            device="cuda",
            loss_function="geometric",
            noise_level=0.1,
        )
        vae_training.train(max_iterations=5000)
        vae_training.fit_gmm(covariance_type="full")


if __name__ == "__main__":
    state_idx_dict = {
        "base_pos": [0, 1, 2],
        "base_quat": [3, 4, 5, 6],
        "base_lin_vel": [7, 8, 9],
        "base_ang_vel": [10, 11, 12],
        "projected_gravity": [13, 14, 15],
        "dof_pos_leg_l": [16, 17, 18, 19, 20],
        "dof_pos_arm_l": [21, 22, 23, 24],
        "dof_pos_leg_r": [25, 26, 27, 28, 29],
        "dof_pos_arm_r": [30, 31, 32, 33],
    }

    """
    Jason 2025-11-10:
    这里估计是取了历史窗口51帧，然后预测未来50帧的状态转移。
    51帧如果是以50Hz采样的话，大概是1秒钟的数据。
    """
    # the window size of the input state transitions
    history_horizon = 51
    # the autoregressive prediction steps while obeying the quasi-constant latent parameterization
    forecast_horizon = 50
    latent_dim = 8

    device = "cuda"
    log_dir_root = LEGGED_GYM_ROOT_DIR + "/logs/flat_mit_humanoid/vae/"
    log_dir = log_dir_root + "misc"
    vae_experiment = VAEExperiment(state_idx_dict, history_horizon, forecast_horizon, device)
    vae_experiment.prepare_data()
    vae_experiment.train(log_dir, latent_dim)
