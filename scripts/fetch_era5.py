import os
import cdsapi
import logging
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_era5_data(
    output_dir: str = "data/raw",
    year: str = "2023",
    month: str = "08", # August is during the Indian Summer Monsoon
    days: list = ["01", "02", "03", "04", "05"],
    area: list = [37.0, 68.0, 8.0, 97.0] # Bounding box for India [N, W, S, E]
):
    """
    Fetches ERA5 hourly data on single levels from the Climate Data Store (CDS).
    Requires a .cdsapirc file setup in the user's home directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = f"era5_india_{year}_{month}_sample.nc"
    filepath = os.path.join(output_dir, filename)

    if os.path.exists(filepath):
        logger.info(f"File {filepath} already exists. Skipping download.")
        return filepath

    logger.info(f"Connecting to CDS API to fetch ERA5 data for {year}-{month}...")
    
    try:
        c = cdsapi.Client()
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'variable': [
                    '2m_temperature',
                    'total_precipitation', # The crucial variable for nowcasting
                    'surface_pressure',
                    '10m_u_component_of_wind',
                    '10m_v_component_of_wind',
                ],
                'year': year,
                'month': month,
                'day': days,
                'time': [f"{str(h).zfill(2)}:00" for h in range(24)],
                'area': area,
                'format': 'netcdf',
            },
            filepath
        )
        logger.info(f"Successfully downloaded ERA5 data to {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Failed to fetch data from CDS API: {e}")
        logger.error("Please ensure you have registered at the Climate Data Store and configured ~/.cdsapirc")
        return None

if __name__ == "__main__":
    fetch_era5_data()
