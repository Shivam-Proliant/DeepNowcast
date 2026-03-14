import torch
import pytest
from src.model import WeatherNowcaster
from src.metrics import critical_success_index, structural_similarity_index

@pytest.fixture
def dummy_tensors():
    # Batch=2, Seq_in=5, Channels=2, Lat=32, Lon=32
    x = torch.randn(2, 5, 2, 32, 32)
    y = torch.randn(2, 2, 2, 32, 32) # Target is 2 hours out
    return x, y

def test_model_forward_shape(dummy_tensors):
    """Ensure the ConvLSTM maps 5 hour inputs to 2 hour predictions accurately in 4D tensor space."""
    x, y = dummy_tensors
    model = WeatherNowcaster(in_channels=2, hidden_dim=8, out_channels=2, seq_out=2)
    
    out = model(x)
    
    # Assert output shape matches expected (Batch, Seq_out, Channels, Lat, Lon)
    assert out.shape == y.shape
    assert out.shape == (2, 2, 2, 32, 32)

def test_critical_success_index():
    # Perfect overlap
    pred = torch.ones(10, 10)
    target = torch.ones(10, 10)
    assert critical_success_index(pred, target) == 1.0
    
    # No overlap
    pred = torch.zeros(10, 10)
    assert critical_success_index(pred, target) == 0.0

def test_ssim():
    pred = torch.ones(1, 1, 1, 32, 32)
    target = torch.ones(1, 1, 1, 32, 32)
    ssim_val = structural_similarity_index(pred, target)
    # Give leniency for floating point math
    assert ssim_val > 0.99
