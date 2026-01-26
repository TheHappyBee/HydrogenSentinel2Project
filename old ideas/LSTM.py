import torch
from torch import nn
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):
    """Basic ConvLSTM cell."""
    def __init__(self, input_channels, hidden_channels, kernel_size, bias=True):
        super().__init__()
        padding = kernel_size // 2
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(input_channels + hidden_channels,
                              4 * hidden_channels,
                              kernel_size,
                              padding=padding,
                              bias=bias)

    def forward(self, x, h, c):
        # x: (B, C, H, W), h/c: (B, hidden, H, W)
        combined = torch.cat([x, h], dim=1)
        conv_out = self.conv(combined)
        (ci, cf, cg, co) = torch.split(conv_out, self.hidden_channels, dim=1)
        i = torch.sigmoid(ci)
        f = torch.sigmoid(cf)
        g = torch.tanh(cg)
        o = torch.sigmoid(co)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTM(nn.Module):
    """Multi-layer ConvLSTM for 2D sequences.

    Input shape: (B, T, C, H, W)
    Returns last hidden state for the top layer by default (B, hidden, H, W)
    """
    def __init__(self, input_channels, hidden_channels, kernel_size=3, num_layers=None):
        super().__init__()
        # Allow num_layers to be inferred from hidden_channels when a list is provided.
        if isinstance(hidden_channels, int):
            # If hidden_channels is an int, require num_layers (default to 1 if None)
            if num_layers is None:
                num_layers = 1
            hidden_channels = [hidden_channels] * num_layers
        else:
            # hidden_channels is a sequence/list. If num_layers is not given, infer it.
            if num_layers is None:
                num_layers = len(hidden_channels)

        assert len(hidden_channels) == num_layers, (
            f'hidden_channels length ({len(hidden_channels)}) must equal num_layers ({num_layers})'
        )
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels

        cells = []
        for i in range(num_layers):
            in_ch = input_channels if i == 0 else hidden_channels[i - 1]
            cells.append(ConvLSTMCell(in_ch, hidden_channels[i], kernel_size))
        self.cells = nn.ModuleList(cells)

    def forward(self, x, hidden=None):
        # x: (B, T, C, H, W)
        b, seq_len, _, h, w = x.size()
        if hidden is None:
            hidden = [(
                torch.zeros(b, ch, h, w, device=x.device),
                torch.zeros(b, ch, h, w, device=x.device),
            ) for ch in self.hidden_channels]

        layer_input = x
        for layer_idx in range(self.num_layers):
            h, c = hidden[layer_idx]
            outputs = []
            cell = self.cells[layer_idx]
            for t in range(seq_len):
                h, c = cell(layer_input[:, t], h, c)
                outputs.append(h)
            layer_input = torch.stack(outputs, dim=1)  # (B, T, hidden, H, W)
            hidden[layer_idx] = (h, c)

        # return full sequence from top layer and final states
        return layer_input, hidden


class ConvLSTMModel(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, num_layers, out_channels):
        super().__init__()
        self.convlstm = ConvLSTM(input_channels, hidden_channels, kernel_size, num_layers)
        top_hidden = hidden_channels if isinstance(hidden_channels, int) else hidden_channels[-1]
        self.conv_out = nn.Conv2d(top_hidden, out_channels, kernel_size=1)

    def forward(self, x, hidden=None):
        # x: (B, T, C, H, W)
        seq_out, hidden = self.convlstm(x, hidden)
        # take last time step output from top layer
        last = seq_out[:, -1]
        out = self.conv_out(last)
        return out, hidden