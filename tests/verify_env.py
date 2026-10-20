"""Environment & Compatibility Verification Script."""

import sys

def verify_all():
    print("Python Executable:", sys.executable)
    print("Python Version:", sys.version)

    import rasterio
    import pystac_client
    import planetary_computer
    import shapely
    import geopandas
    import torch
    import torchvision
    import fastapi

    print("\n--- Package Verification ---")
    print(f"rasterio: {rasterio.__version__} (GDAL: {rasterio.__gdal_version__})")
    print(f"pystac-client: {pystac_client.__version__}")
    print(f"planetary-computer: {planetary_computer.__version__}")
    print(f"shapely: {shapely.__version__}")
    print(f"geopandas: {geopandas.__version__}")
    print(f"torch: {torch.__version__}")
    print(f"torchvision: {torchvision.__version__}")
    print(f"fastapi: {fastapi.__version__}")

    # Verify rasterio driver capabilities
    with rasterio.Env() as env:
        drivers = env.drivers()
        print("\n--- GDAL Driver Verification ---")
        print(f"Total GDAL Drivers Available: {len(drivers)}")
        for drv in ["GTiff", "PNG", "JPEG", "VRT"]:
            print(f"Driver '{drv}': {'Supported' if drv in drivers else 'Missing'}")

    print("\nEnvironment compatibility check passed successfully!")

if __name__ == "__main__":
    verify_all()
