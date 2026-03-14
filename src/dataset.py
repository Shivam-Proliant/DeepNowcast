import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np
import logging
from typing import Tuple, List

logger = logging.getLogger(__name__)

class WeatherDataset(Dataset):
    """
    Spatiotemporal Dataset wrapper for ERA5/IMD NetCDF files.
    Treats meteorological forecasting as a video prediction problem.
    """
    def __init__(self, nc_file_path: str, seq_in: int = 5, seq_out: int = 2, variables: List[str] = None):
        """
        Args:
            nc_file_path: Path to the .nc file.
            seq_in: Number of past hours to input (default 5).
            seq_out: Number of future hours to predict (default 2).
            variables: List of NetCDF variable names to extract as channels.
        """
        self.seq_in = seq_in
        self.seq_out = seq_out
        self.total_seq_len = seq_in + seq_out
        
        if variables is None:
            self.variables = ['t2m', 'tp'] # default: 2m temp and total precipitation
        else:
            self.variables = variables

        logger.info(f"Loading Dataset from {nc_file_path} for variables {self.variables}...")
        try:
            # Load dataset using xarray
            self.ds = xr.open_dataset(nc_file_path)
            
            # Extract underlying numpy arrays
            data_vars = [self.ds[var].values for var in self.variables]
            
            # Stack into shape: (Time, Channels, Lat, Lon)
            self.data = np.stack(data_vars, axis=1)
            
            # Replace NaNs (often found over oceans or masked areas) with 0
            self.data = np.nan_to_num(self.data, nan=0.0)
            
            self.num_samples = len(self.data) - self.total_seq_len + 1
            
            # Compute basic Min-Max normalization per channel globally
            self._normalize()
            logger.info(f"Dataset Loaded. Shape: {self.data.shape}, Total Samples: {self.num_samples}")

        except Exception as e:
            logger.error(f"Failed to load NetCDF file: {e}")
            raise e

    def _normalize(self):
        """Normalizes each weather variable (channel) to [0, 1] range."""
        self.mins = np.min(self.data, axis=(0, 2, 3), keepdims=True)
        self.maxs = np.max(self.data, axis=(0, 2, 3), keepdims=True)
        # Avoid division by zero
        range_vals = np.where((self.maxs - self.mins) == 0, 1e-5, (self.maxs - self.mins))
        self.data = (self.data - self.mins) / range_vals

    def __len__(self) -> int:
        return max(0, self.num_samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            X: Input tensor (seq_in, Channels, Lat, Lon)
            Y: Target tensor (seq_out, Channels, Lat, Lon)
        """
        # Slicing the continuous time-series
        sequence = self.data[idx : idx + self.total_seq_len]
        
        x_data = sequence[:self.seq_in]
        y_data = sequence[self.seq_in:]
        
        return torch.FloatTensor(x_data), torch.FloatTensor(y_data)

# Example Usage
if __name__ == "__main__":
    # Create dummy data if no NC file exists to verify shape logic
    dummy_nc_path = "dummy.nc"
    if not __import__('os').path.exists(dummy_nc_path):
        import pandas as pd
        times = pd.date_range("2023-01-01", periods=100, freq="1H")
        lats = np.linspace(37.0, 8.0, 32)
        lons = np.linspace(68.0, 97.0, 32)
        
        temp = np.random.rand(100, 32, 32)
        precip = np.random.rand(100, 32, 32)
        
        ds = xr.Dataset(
            {
                "t2m": (["time", "latitude", "longitude"], temp),
                "tp": (["time", "latitude", "longitude"], precip),
            },
            coords={
                "longitude": lons,
                "latitude": lats,
                "time": times,
            },
        )
        ds.to_netcdf(dummy_nc_path)
        
    dataset = WeatherDataset(dummy_nc_path, seq_in=5, seq_out=2)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    x, y = next(iter(dataloader))
    print(f"Batch X shape: {x.shape} -> (Batch, Seq_in, Channels, Lat, Lon)")
    print(f"Batch Y shape: {y.shape} -> (Batch, Seq_out, Channels, Lat, Lon)")