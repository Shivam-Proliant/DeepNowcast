import torch
import torch.nn as nn
from typing import Tuple

class ConvLSTMCell(nn.Module):
    """
    A single Convolutional LSTM Cell. 
    Replaces fully-connected matrix multiplications with convolutions to preserve 
    spatial features (e.g., weather fronts, cyclones) while modeling time sequence.
    """
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int, bias: bool = True):
        super(ConvLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        
        # Convolutions for all 4 LSTM gates (Input, Forget, Output, Gate)
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim, 
            kernel_size=kernel_size,
            padding=padding,
            bias=bias
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        # Concatenate spatial input with previous spatial hidden state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

class WeatherNowcaster(nn.Module):
    """
    Spatiotemporal Deep Learning Architecture for Weather Nowcasting.
    Treats forecasting as a Video Prediction problem (Seq2Seq architecture).
    """
    def __init__(self, in_channels: int, hidden_dim: int, out_channels: int, seq_out: int):
        """
        Args:
            in_channels: Number of input weather variables (e.g., temp, precip)
            hidden_dim: Number of hidden channels in the ConvLSTM
            out_channels: Number of output weather variables to predict
            seq_out: Number of future hours/frames to output
        """
        super(WeatherNowcaster, self).__init__()
        self.seq_out = seq_out
        self.hidden_dim = hidden_dim
        
        # Encoder: Extracts spatial and temporal features
        self.convlstm = ConvLSTMCell(input_dim=in_channels, hidden_dim=hidden_dim, kernel_size=3)
        
        # Decoder: Maps the high-dimensional hidden state back to the target weather variables
        self.conv_out = nn.Conv2d(in_channels=hidden_dim, out_channels=out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward Pass for Seq2Seq Nowcasting.
        Args:
            x: Input tensor of shape (Batch, Seq_in, Channels, Lat, Lon)
               E.g., 5 past hours of temperature and rainfall for a 64x64 grid.
        Returns:
            predictions: Tensor of shape (Batch, Seq_out, Channels, Lat, Lon)
        """
        b, seq_len, _, h, w = x.size()
        
        # Initialize hidden states
        h_t = torch.zeros(b, self.hidden_dim, h, w).to(x.device)
        c_t = torch.zeros(b, self.hidden_dim, h, w).to(x.device)
        
        # Encode the past frames
        for t in range(seq_len):
            h_t, c_t = self.convlstm(x[:, t, :, :, :], (h_t, c_t))
            
        # Predict future frames autoregressively
        predictions =[]
        for _ in range(self.seq_out):
            pred = self.conv_out(h_t)
            predictions.append(pred)
            # Feed prediction back into the cell for the next step
            h_t, c_t = self.convlstm(pred, (h_t, c_t))
            
        # Output shape: (Batch, Seq_out, Channels, Lat, Lon)
        return torch.stack(predictions, dim=1)
