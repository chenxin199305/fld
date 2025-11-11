import torch


class DistributionBuffer:
    """
    A class to manage a circular buffer for storing distributions.

    Attributes:
        distribution_buffer (torch.Tensor): A tensor to store the buffer data.
        buffer_size (int): The maximum size of the buffer.
        device (torch.device): The device on which the buffer is stored.
        step (int): The current position in the buffer for the next insertion.
        num_samples (int): The number of samples currently in the buffer.
    """

    def __init__(self, buffer_dim, buffer_size, device: str) -> None:
        """
        Initializes the DistributionBuffer.

        Args:
            buffer_dim (int): The dimensionality of the buffer.
            buffer_size (int): The maximum number of elements the buffer can hold.
            device (torch.device): The device on which the buffer will be stored.
        """
        self.distribution_buffer = torch.zeros(buffer_size, buffer_dim, dtype=torch.float, requires_grad=False).to(device)
        self.buffer_size = buffer_size
        self.device = device
        self.step = 0
        self.num_samples = 0

    def insert(self, data):
        """
        Inserts new data into the buffer. Overwrites old data if the buffer is full.

        Args:
            data (torch.Tensor): A tensor containing the data to insert. The first dimension
                                 represents the number of data points.
        """

        print(
            f"[DistributionBuffer] \n"
            f"Inserting new data into DistributionBuffer...\n"
            f"data.shape: {data.shape}\n"
        )

        num_data = data.shape[0]
        start_idx = self.step
        end_idx = self.step + num_data

        if end_idx > self.buffer_size:
            # Handle wrap-around case
            self.distribution_buffer[self.step:self.buffer_size] = data[:self.buffer_size - self.step]
            self.distribution_buffer[:end_idx - self.buffer_size] = data[self.buffer_size - self.step:]
        else:
            # Insert data without wrap-around
            self.distribution_buffer[start_idx:end_idx] = data

        # Update the number of samples and the step index
        self.num_samples = min(self.buffer_size, max(end_idx, self.num_samples))
        self.step = (self.step + num_data) % self.buffer_size

    def get_distribution(self):
        """
        Retrieves the current state of the distribution buffer.

        Returns:
            torch.Tensor: The tensor representing the distribution buffer.
        """
        return self.distribution_buffer
