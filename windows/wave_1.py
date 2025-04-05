import torch
import torch.nn as nn

# Define a custom 1D causal convolution layer
class CausalConv1d(nn.Conv1d):
    """1D Causal convolution layer that pads inputs to avoid using future data."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        # Compute padding to ensure output length matches input length
        padding = (kernel_size - 1) * dilation
        super().__init__(in_channels, out_channels, kernel_size,
                         padding=padding, dilation=dilation, **kwargs)

    def forward(self, x):
        out = super().forward(x)
        if self.padding[0] > 0:
            out = out[:, :, :-self.padding[0]]
        return out

class WaveNetForecaster(nn.Module):
    def __init__(self, in_channels=32, residual_channels=32, skip_channels=64,
                 kernel_size=2, num_layers=8):
        """
        WaveNet-based forecaster model.
        Args:
            in_channels: Number of input channels.
            residual_channels: Number of channels in the residual layers.
            skip_channels: Number of channels in the skip connections.
            kernel_size: Size of the convolutional kernel.
            num_layers: Number of dilated causal convolution layers.
        """
        super(WaveNetForecaster, self).__init__()
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels

        # Initial 1x1 convolution to project input to residual_channels
        self.input_conv = nn.Conv1d(in_channels, residual_channels, kernel_size=1)

        # Lists to hold the layers for each dilated block
        self.filter_convs = nn.ModuleList()  # Convolutions for the filter branch
        self.gate_convs = nn.ModuleList()    # Convolutions for the gate branch
        self.residual_convs = nn.ModuleList()  # Residual connections
        self.skip_convs = nn.ModuleList()      # Skip connections

        # Create layers with exponentially increasing dilation rates
        for i in range(num_layers):
            dilation = 2 ** i  # Dilation rate doubles at each layer
            # Add causal convolutions for filter and gate branches
            self.filter_convs.append(CausalConv1d(residual_channels, residual_channels,
                                                  kernel_size, dilation=dilation))
            self.gate_convs.append(CausalConv1d(residual_channels, residual_channels,
                                                kernel_size, dilation=dilation))
            # Add 1x1 convolutions for residual and skip connections
            self.residual_convs.append(nn.Conv1d(residual_channels, residual_channels, kernel_size=1))
            self.skip_convs.append(nn.Conv1d(residual_channels, skip_channels, kernel_size=1))

        # Output layers to process accumulated skip connections
        self.output_conv1 = nn.Conv1d(skip_channels, skip_channels, kernel_size=1)
        self.output_conv2 = nn.Conv1d(skip_channels, in_channels, kernel_size=1)  # Map back to input channels

    def forward(self, x):
        """
        Forward pass of the WaveNet model.
        Args:
            x: Input tensor of shape [batch, in_channels, input_length].
        Returns:
            Output tensor of shape [batch, in_channels, input_length].
        """
        # Project input to residual_channels using the initial 1x1 convolution
        x = self.input_conv(x)
        skip_sum = None  # Initialize skip connection accumulator

        # Pass through the stack of dilated causal layers
        for filter_conv, gate_conv, res_conv, skip_conv in zip(
                self.filter_convs, self.gate_convs, self.residual_convs, self.skip_convs):
            # Compute gated activation unit (element-wise multiplication of tanh and sigmoid outputs)
            filt = torch.tanh(filter_conv(x))
            gate = torch.sigmoid(gate_conv(x))
            out = filt * gate

            # Compute skip connection output and accumulate it
            skip_out = skip_conv(out)
            skip_sum = skip_out if skip_sum is None else (skip_sum + skip_out)

            # Compute residual connection and add it to the input for the next layer
            x = res_conv(out) + x

        # Process the accumulated skip connections through output layers
        out = torch.relu(skip_sum)
        out = torch.relu(self.output_conv1(out))
        out = self.output_conv2(out)

        # Return the final output tensor
        return out
