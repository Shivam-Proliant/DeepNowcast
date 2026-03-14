import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import logging
from src.model import WeatherNowcaster
from src.metrics import critical_success_index, structural_similarity_index
from src.dataset import WeatherDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model(config_path: str = "config.yaml"):
    # Load Configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize Model 
    model = WeatherNowcaster(
        in_channels=config['model']['in_channels'], 
        hidden_dim=config['model']['hidden_dim'], 
        out_channels=config['model']['out_channels'], 
        seq_out=config['training']['seq_out']
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    # Initialize DataLoader
    try:
        dataset = WeatherDataset(
            nc_file_path=config['data']['era5_nc_path'],
            seq_in=config['training']['seq_in'],
            seq_out=config['training']['seq_out'],
            variables=config['data']['variables']
        )
        dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)
        use_dummy_data = False
        logger.info("Successfully loaded ERA5 training dataloader.")
    except Exception as e:
        logger.warning(f"Could not load real data fallback to dummy tensors. Error: {e}")
        use_dummy_data = True
        dummy_x = torch.randn(config['training']['batch_size'], config['training']['seq_in'], config['model']['in_channels'], 64, 64).to(device)
        dummy_y = torch.randn(config['training']['batch_size'], config['training']['seq_out'], config['model']['out_channels'], 64, 64).to(device)
    
    model.train()
    epochs = config['training']['epochs']
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_csi = 0.0
        epoch_ssim = 0.0
        batches = 0
        
        if use_dummy_data:
            # Dummy logic
            optimizer.zero_grad()
            outputs = model(dummy_x)
            loss = criterion(outputs, dummy_y)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss = loss.item()
            epoch_csi = critical_success_index(outputs, dummy_y)
            epoch_ssim = structural_similarity_index(outputs, dummy_y)
            batches = 1
        else:
            for x_batch, y_batch in dataloader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                
                optimizer.zero_grad()
                outputs = model(x_batch)
                
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_csi += critical_success_index(outputs, y_batch)
                epoch_ssim += structural_similarity_index(outputs, y_batch)
                batches += 1
                
        logger.info(f"Epoch[{epoch+1}/{epochs}], Loss: {epoch_loss/batches:.4f}, CSI: {epoch_csi/batches:.4f}, SSIM: {epoch_ssim/batches:.4f}")

    # Save artifact for MLOps phase
    save_path = config['training']['save_path']
    torch.save(model.state_dict(), save_path)
    logger.info(f"Model saved to {save_path}")
    # Dummy Data for boilerplate compilation: (Batch, Seq, Channels, Lat, Lon)
    dummy_x = torch.randn(8, 5, 2, 64, 64).to(device)
    dummy_y = torch.randn(8, 2, 2, 64, 64).to(device)
    
    model.train()
    for epoch in range(10): # 10 epochs for demo
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(dummy_x)
        
        # Calculate Loss & CSI
        loss = criterion(outputs, dummy_y)
        csi_score = critical_success_index(outputs, dummy_y)
        
        # Backprop
        loss.backward()
        optimizer.step()
        
        print(f"Epoch[{epoch+1}/10], Loss: {loss.item():.4f}, CSI: {csi_score:.4f}")

    # Save artifact for MLOps phase
    torch.save(model.state_dict(), "deepnowcast_v1.pth")

if __name__ == "__main__":
    train_model()
