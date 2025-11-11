import torch
import numpy as np


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, obs_dim, obs_horizon, buffer_size, device):
        """
        Initialize a ReplayBuffer object.

        Arguments:
            buffer_size (int): maximum size of buffer
        """
        self.state_buf = torch.zeros(buffer_size, obs_horizon, obs_dim).to(device)
        self.buffer_size = buffer_size
        self.device = device

        self.step = 0
        self.num_samples = 0

    def insert(self, state_buf):
        """
        Add new states to memory.
        """

        num_states = state_buf.shape[0]
        start_idx = self.step
        end_idx = self.step + num_states

        if end_idx > self.buffer_size:
            self.state_buf[self.step:self.buffer_size] = state_buf[:self.buffer_size - self.step]
            self.state_buf[:end_idx - self.buffer_size] = state_buf[self.buffer_size - self.step:]
        else:
            self.state_buf[start_idx:end_idx] = state_buf

        self.num_samples = min(self.buffer_size, max(end_idx, self.num_samples))
        self.step = (self.step + num_states) % self.buffer_size

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        """
        Yield mini-batches of experiences.

        Args:
            num_mini_batch (int): Number of mini-batches to generate.
            mini_batch_size (int): Size of each mini-batch.
        """
        for _ in range(num_mini_batch):
            # 随机选择 mini_batch_size 个样本的索引
            sample_idxs = np.random.choice(self.num_samples, size=mini_batch_size)

            """
            Jason 2025-11-10:
            所谓 yield, 就是一个生成器的意思。调用这个函数的时候，并不会立刻执行函数体内的代码，而是返回一个生成器对象。
            当你对这个生成器对象进行迭代时，函数体内的代码才会被执行，直到遇到 yield 语句时，函数会暂停执行，并返回 yield 后面的值。
            下一次迭代时，函数会从暂停的地方继续执行，直到再次遇到 yield 语句，或者函数执行完毕。
            这样可以节省内存，因为不需要一次性生成所有的值，而是按需生成。
            这里的 yield 语句返回了一个 mini-batch 的状态数据。
            这个 mini_batch 的状态数据是从 self.state_buf 中随机抽取的，大小为 mini_batch_size。
            通过这种方式，可以在训练过程中动态地获取 mini-batch 数据，而不需要预先存储所有的 mini-batch。
            """
            yield self.state_buf[sample_idxs, :].to(self.device)
