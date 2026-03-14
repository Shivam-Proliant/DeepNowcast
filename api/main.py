from fastapi import FastAPI, UploadFile, File, Request
import torch
import numpy as np
from src.model import WeatherNowcaster

app = FastAPI(title="DeepNowcast API", description="Real-time Spatiotemporal Weather Nowcasting")

# Load model globally on startup (Mocked values matching config.yaml)
device = torch.device('cpu')
model = WeatherNowcaster(in_channels=2, hidden_dim=32, out_channels=2, seq_out=2)
# In a real environment, load via OS paths to the .pth artifact
# model.load_state_dict(torch.load('deepnowcast_v1.pth', map_location=device))
model.eval()

@app.post("/predict")
async def predict_nowcast(request: Request):
    """
    Endpoint receives a preprocessed tensor via JSON (e.g., from a live API scraper).
    Expects a nested list representing past 5 hours of weather grids: [1, 5, 2, Lat, Lon].
    """
    try:
        data = await request.json()
        tensor_list = data.get("tensor_data")
        
        if tensor_list is None:
            # Fallback for boilerplate/demo testing without passing massive JSON
            tensor_data = torch.randn(1, 5, 2, 32, 32)
        else:
            numpy_data = np.array(tensor_list, dtype=np.float32)
            tensor_data = torch.from_numpy(numpy_data)
            
        with torch.no_grad():
            prediction = model(tensor_data)
            
        return {
            "status": "success",
            "message": "Future 2-hour spatial grids generated successfully.",
            "predicted_shape": list(prediction.shape),
            "data_sample": prediction[0, 0, 0, 0, :5].tolist() # Sending a tiny snippet for the JSON response
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
    return {
        "status": "success",
        "predicted_frames": prediction.shape[1],
        "message": "Future 2-hour spatial grids generated successfully.",
        "data_sample": prediction[0, 0, 0, 0, :5].tolist() # Sending a tiny snippet for the JSON response
    }
