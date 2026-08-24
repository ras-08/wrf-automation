import os
import argparse
import numpy as np
import pandas as pd
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from wrf import getvar, ll_to_xy, to_np
import metpy.calc as mpcalc
from metpy.plots import SkewT, add_metpy_logo
from metpy.units import units
from datetime import datetime, timedelta
import warnings
# Suppress the xarray FutureWarnings to keep your terminal clean
warnings.filterwarnings("ignore", category=FutureWarning)

# --- 1. EDITABLE DICTIONARY ---
# Add or delete cities here as needed
locations = {
    "Kerala": {"lat": 10, "lon": 76},
    "Mumbai": {"lat": 19.0760, "lon": 72.8777},
    # "Mombasa": {"lat": -4.0435, "lon": 39.6682},
    # "Nakuru": {"lat": -0.3031, "lon": 36.0800},
}

# Path to your WRF output file
# WRF_FILE = "/home/erickwambugu/WRF/run/wrfout_d01_2026-06-03_00:00:00"

def generate_all_skewts(wrf_path, city_dict):
    ncfile = Dataset(wrf_path)
    
    # Extract date for folder naming (e.g., 20260601)
    wrf_time_raw = getvar(ncfile, "times", timeidx=0)
    date_folder = pd.to_datetime(str(wrf_time_raw.values)).strftime('%Y%m%d')
    
    # Get total time steps
    ntimes = ncfile.dimensions['Time'].size
    
    # Loop through each city in the dictionary
    for city_name, coords in city_dict.items():
        print(f"\nProcessing profiles for: {city_name}")
        
        # Create specific folder: /home/erickwambugu/WEATHER/INDIA/Data/YYYYMMDD/skewplots/CityName
        output_dir = f"/home/ras_08/WEATHER/INDIA/Data/{date_folder}/skewplots/{city_name}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Find grid points
        x_y = ll_to_xy(ncfile, coords['lat'], coords['lon'])
        
        for t in range(ntimes):
            # Extract vertical profile data
            p = units.Quantity(to_np(getvar(ncfile, "pressure", timeidx=t)[:, x_y[1], x_y[0]]), "hPa")
            tc = units.Quantity(to_np(getvar(ncfile, "tc", timeidx=t)[:, x_y[1], x_y[0]]), "degC")
            td = units.Quantity(to_np(getvar(ncfile, "td", timeidx=t)[:, x_y[1], x_y[0]]), "degC")
            u = units.Quantity(to_np(getvar(ncfile, "ua", timeidx=t)[:, x_y[1], x_y[0]]), "m/s")
            v = units.Quantity(to_np(getvar(ncfile, "va", timeidx=t)[:, x_y[1], x_y[0]]), "m/s")

            # Timestamp for the title
            valid_time = pd.to_datetime(str(getvar(ncfile, "times", timeidx=t).values))
            eat_time = valid_time + timedelta(hours=0)  # Convert UTC to EAT
            eat_time_str = eat_time.strftime('%Y-%m-%d | %H:%M UTC')

            # --- PLOTTING ---
            fig = plt.figure(figsize=(10, 15))
            skew = SkewT(fig, rotation=45)

            skew.plot(p, tc, 'r', linewidth=2, label='Temp')
            skew.plot(p, td, 'g', linewidth=2, label='Dewpoint')
            
            # Resample wind barbs for clarity
            levels_to_plot = np.arange(100, 1050, 50) * units.hPa
            idx = mpcalc.resample_nn_1d(p, levels_to_plot)
            skew.plot_barbs(p[idx], u[idx], v[idx])

            lcl_pressure, lcl_temperature = mpcalc.lcl(p[0], tc[0], td[0])
            skew.plot(lcl_pressure, lcl_temperature, 'ko', markerfacecolor='black', label='LCL')

            prof = mpcalc.parcel_profile(p, tc[0], td[0]).to('degC')
            skew.plot(p, prof, 'k--', linewidth=2, label='Parcel Path')

            skew.shade_cin(p, tc, prof, td)
            skew.shade_cape(p, tc, prof)

            skew.ax.axvline(0, color='c', linestyle='--', linewidth=2)  


            # Reference lines
            skew.plot_dry_adiabats()
            skew.plot_moist_adiabats()
            skew.plot_mixing_lines()
            
            skew.ax.set_ylim(1050, 100)
            skew.ax.set_xlim(-40, 60)

            skew.ax.set_xlabel("Temperature (°C)", fontsize=10)
            skew.ax.set_ylabel("Pressure (hPa)", fontsize=10)
            skew.ax.legend(loc='upper right', fontsize=8)
            
            plt.title(f"Skew-T Log-P Analysis For: {city_name}\nValid:{eat_time_str}", 
                      fontsize=12, fontweight='bold')
            
            # Save using city name and timestep
            save_path = os.path.join(output_dir, f"{city_name}_skewT-Time:{eat_time_str}.png")
            plt.savefig(save_path, dpi=120, bbox_inches='tight')
            plt.close(fig)
            
        print(f"Done. 24 plots saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    generate_all_skewts(args.input, locations)
