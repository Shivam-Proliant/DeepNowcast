DeepNowcast: Spatiotemporal Deep Learning for Weather Nowcasting

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)

**DeepNowcast** is an end-to-end Machine Learning pipeline designed to predict highly localized, short-term weather events (0–6 hours) such as the Indian Summer Monsoon (ISMR) and severe cyclones. 

> **Statement of Purpose Context**: While working as an R&D Systems Administrator, I became fascinated by how infrastructure and AI intersect. Traditional Numerical Weather Prediction (NWP) models take hours to compute on superclusters. I built *DeepNowcast* to test the hypothesis that we can treat weather forecasting as a **video prediction problem**, utilizing GPU-accelerated **ConvLSTM** networks to achieve real-time inference. This project solidifies my desire to pursue an MS in AI to deepen my algorithmic knowledge.

---
 Architecture & Approach

This project moves beyond standard 2D CSV data into complex **4D Scientific Tensors** `(Time x Channels x Lat x Lon)`. 

1. **The Data Pipeline (SysAdmin/Data Eng)**: Automates fetching ERA5 Reanalysis data (NetCDF4 format) via the Climate Data Store (CDS) API.
2. **Spatiotemporal Modeling (ConvLSTM)**: Replaces standard fully-connected LSTM gates with convolutions. This preserves spatial features (like a rotating storm front) moving across time.
3. **Advanced Evaluation**: Standard accuracy is meaningless for rare extreme weather. DeepNowcast evaluates predictions using the **Critical Success Index (CSI)** and **Structural Similarity Index (SSIM)**.
4. **MLOps Deployment**: Containerized with Docker and surfaced via a FastAPI inference endpoint.

---

 Quickstart

### 1. Installation
Clone the repo and configure your environment:
```bash
git clone https://github.com/Shivam-Proliant/project/DeepNowcast.git
cd DeepNowcast
pip install -r requirements.txt
```

### 2. Configure CDS API
To download real ERA5 NetCDF data, register at the [Climate Data Store](https://cds.climate.copernicus.eu/) and place your `.cdsapirc` file in your home directory (`~/.cdsapirc`).

### 3. Fetch Data & Train
```bash
# Fetch 5 days of hourly ERA5 chunks for the Indian bounding box
python scripts/fetch_era5.py

# Train the ConvLSTM (reads from config.yaml)
python src/train.py
```

### 4. Deploy the API via Docker
```bash
docker build -t deepnowcast .
docker run -p 8000:8000 deepnowcast
```
You can now send simulated inference tensors to `http://localhost:8000/predict`.

---
 Testing (CI/CD)

The repository uses `pytest` for unit testing model geometries and validation metrics. A GitHub Actions workflow runs on every push:

```bash
pytest tests/
flake8 src/ api/
```

 Future Work
For an MS thesis or further development, this architecture can be upgraded to use **Vision Transformers (ViT)** or Graph Neural Networks (resembling DeepMind's GraphCast).

## License
MIT License
