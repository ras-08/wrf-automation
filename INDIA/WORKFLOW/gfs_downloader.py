import os
import requests
import argparse
from datetime import datetime
from pathlib import Path

def download_gfs_india(date_str, output_path, cycle='00', forecast_hours=range(0, 27, 3)):
    """
    Downloads GFS data for the India bounding box.
    """
    base_url = 'https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl'

    # India bounding box
    toplat = 55
    bottomlat = -10
    leftlon = 40
    rightlon = 120

    # Ensure the directory exists
    save_dir = Path(output_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    data_dir = f'/gfs.{date_str}/{cycle}/atmos'
    downloaded_files = []
    failed_files = []

    print(f"--- Starting download for {date_str} to {save_dir} ---")

    for fhr in forecast_hours:
        fhr_str = f'{fhr:03d}'
        file_name = f'gfs.t{cycle}z.pgrb2.0p25.f{fhr_str}'
        
        params = {
            'dir': data_dir,
            'file': file_name,
            'all_var': 'on',
            'all_lev': 'on',
            'subregion': '',
            'toplat': toplat,
            'leftlon': leftlon,
            'rightlon': rightlon,
            'bottomlat': bottomlat,
        }

        try:
            response = requests.get(base_url, params=params, timeout=120)
            if response.status_code == 200:
                output_file = save_dir / f'{file_name}.grib2'
                with open(output_file, 'wb') as f:
                    f.write(response.content)
                downloaded_files.append(str(output_file))
                print(f"Successfully downloaded: {file_name}")
            else:
                failed_files.append(file_name)
                print(f"Failed (Status {response.status_code}): {file_name}")
        except Exception as e:
            failed_files.append(file_name)
            print(f"Error downloading {file_name}: {e}")

    return len(downloaded_files) > 0

if __name__ == "__main__":
    # Set up Command Line Arguments
    parser = argparse.ArgumentParser(description="Download GFS data for India WRF runs.")
    
    datetime_date = datetime.utcnow().strftime('%Y%m%d')

    parser.add_argument("--date", type=str, help="Date in YYYYMMDD format", 
                        default=datetime_date)
    parser.add_argument("--outdir", type=str, required=True, 
                        help="Full path to the directory where data should be saved")
    
    args = parser.parse_args()

    # Execute the function
    success = download_gfs_india(date_str=args.date, output_path=args.outdir)
    
    if success:
        print("GFS Download Complete.")
    else:
        print("GFS Download Failed or partially failed.")
