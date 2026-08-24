import os
import argparse
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.animation import FFMpegWriter
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from netCDF4 import Dataset
from wrf import getvar, latlon_coords, to_np
import pandas as pd
from datetime import datetime, timedelta

def run_rainfall_pipeline(wrf_path, output_dir):
    # --- 1. SETUP & DATA LOADING ---
    ncfile = Dataset(wrf_path)
    shapefile_path = "/home/ras_08/WEATHER/INDIA/WORKFLOW/map_json_india/india_found/Admin2.shp"
    counties = gpd.read_file(shapefile_path)
    states = counties.dissolve(by="ST_NM")
    label = str(states)
    
    colors = [
         "#ffff99",  # 1-2.5
         "#CECE0C",  # 2.5-10
         "#0BFF0B",  # 10-20
         "#1B5E20", # IMD-like dark green,  # 20-40
         "#00B7FF",  # 40-70
         "#0A2683",  # 70-130
         "#F3931D",  # 130-200  <-- ORANGE
         "#FF0000",  # >200     <-- RED
         ]


    levels = [1, 2.5, 10, 20, 40, 70, 130, 200,300]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(levels, cmap.N)

    # --- 2. COORDINATE & TIME PREP ---
    rain_start_var = getvar(ncfile, "RAINC", timeidx=0) + getvar(ncfile, "RAINNC", timeidx=0)
    lats, lons = latlon_coords(rain_start_var)
    lon_min, lon_max = to_np(lons).min(), to_np(lons).max()
    lat_min, lat_max = to_np(lats).min(), to_np(lats).max()

    def format_eat_time(wrf_time_raw):
        """Converts WRF UTC string to EAT (UTC+0) string."""
        ts = pd.to_datetime(str(wrf_time_raw))
        eat_time = ts + timedelta(hours=0)
        return eat_time.strftime('%d %b %Y | %H:%M UTC')

    def apply_map_features(axis):
        axis.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        # axis.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
        axis.add_feature(cfeature.COASTLINE, linewidth=2, zorder=4)
        axis.add_feature(cfeature.LAND, facecolor="white", zorder=0)
        # axis.add_feature(cfeature.LAKES, facecolor="#0974b3", edgecolor="blue", alpha=0.3, zorder=1)
        # axis.add_feature(cfeature.BORDERS, edgecolor="black", linewidth=2, zorder=4)
        # axis.add_feature(cfeature.RIVERS, facecolor="#0974b3", edgecolor="blue", alpha=0.3, zorder=1)


        # District boundaries
        counties.boundary.plot(ax=axis,edgecolor="#FF00FF",linewidth=0.2,zorder=4)

        # State boundaries
        states.boundary.plot(ax=axis,edgecolor="#FF00FF",linewidth=1.2,zorder=5)
        for state_name, row in states.iterrows():
            point = row.geometry.representative_point()

            axis.text(
                point.x,
                point.y,
                state_name,
                transform=ccrs.PlateCarree(),
                fontsize=6,
                color="#FF00FF",
                fontweight="bold",
                ha="center",
                va="center",
                zorder=10
                )

        '''for idx, row in states.iterrows():
            if 'NAME_LANG1' in row and pd.notnull(row['NAME_LANG1']):
                axis.text(
                row.geometry.centroid.x, 
                row.geometry.centroid.y, 
                row['NAME_LANG1'], 
                transform=ccrs.PlateCarree(), 
                fontsize=6, 
                fontstyle='italic',
                fontweight='bold', 
                ha='center',
                zorder=4
                )'''
        

        # # Label Neighbors
        # style = {
        #     'transform': ccrs.PlateCarree(), 
        #     'fontsize': 12, 
        #     'fontweight': 'bold', 
        #     'ha': 'center', 
        #     'zorder': 7
        # }

        # axis.text(38.0, 0.5, "KENYA", **style)
        # axis.text(42.0, 2.0, "SOMALIA", **style)
        # axis.text(39.0, 5.0, "ETHIOPIA", **style)
        # axis.text(34.0, 2.0, "UGANDA", **style)
        # axis.text(35.0, -3.0, "TANZANIA", **style)
        # axis.text(34.0, 4.8, "SOUTH SUDAN", **style)
        # axis.text(41.5, -3.0, "INDIAN OCEAN", **style)


        gl = axis.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, 
                            color='gray', alpha=0.3, linestyle='--')
        
        gl.top_labels = gl.right_labels = False
        gl.xformatter, gl.yformatter = LONGITUDE_FORMATTER, LATITUDE_FORMATTER
        gl.xlabel_style = {'fontsize': 10, 'color': 'black'}
        gl.ylabel_style = {'fontsize': 10, 'color': 'black'}

    # --- 3. CUMULATIVE 24-HOUR PLOT ---
    print("Generating 24-hour Cumulative Rainfall Map...")
    fig = plt.figure(figsize=(12, 15))
    ax = plt.axes(projection=ccrs.PlateCarree())
    apply_map_features(ax)
    
    rain_end = getvar(ncfile, "RAINC", timeidx=-1) + getvar(ncfile, "RAINNC", timeidx=-1)
    rain_total = rain_end - rain_start_var
    rain_total = np.ma.masked_less(to_np(rain_total), 1.0)
    
    contour = ax.contourf(to_np(lons), to_np(lats), to_np(rain_total),
                          levels=levels, cmap=cmap, norm=norm, extend="max", 
                          transform=ccrs.PlateCarree(), zorder=2, alpha=0.9)
    
    plt.colorbar(contour, ax=ax, orientation="vertical", pad=0.02, shrink=0.7, 
                 label="Total Rainfall Accumulation (mm)")
    
    start_time_eat = format_eat_time(getvar(ncfile, "Times", timeidx=0).values)
    end_time_eat = format_eat_time(getvar(ncfile, "Times", timeidx=-1).values)
    
    plt.title(f"NWP 24-Hour Cumulative Rainfall\nValid: {start_time_eat} <--> {end_time_eat}", 
              fontsize=14, fontweight="bold")
    plt.savefig(os.path.join(output_dir, "cumulative_rainfall.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # --- 4. HOURLY ANIMATION ---
    print("Generating slow-motion animation...")
    fig, ax = plt.subplots(figsize=(12, 15), subplot_kw={'projection': ccrs.PlateCarree()})
    apply_map_features(ax)
    
    # Static Colorbar for Animation
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, ax=ax, orientation="horizontal", pad=0.03, shrink=0.8, label="Hourly Rainfall (mm)")
    
    ntimes = ncfile.dimensions['Time'].size

    def update(frame):
        for c in ax.collections:
            if hasattr(c, "get_transform") and getattr(c, 'zorder', 0) == 3:
                c.remove()
        
        r_prev = getvar(ncfile, "RAINC", timeidx=frame) + getvar(ncfile, "RAINNC", timeidx=frame)
        r_now = getvar(ncfile, "RAINC", timeidx=frame+1) + getvar(ncfile, "RAINNC", timeidx=frame+1)
        hourly = to_np(r_now - r_prev)
        hourly = np.ma.masked_less_equal(hourly, 0.1)
        
        new_plot = ax.contourf(to_np(lons), to_np(lats), hourly,
                               levels=levels, cmap=cmap, norm=norm, extend="max", 
                               transform=ccrs.PlateCarree(), zorder=2, alpha=0.8)
        
        current_time_eat = format_eat_time(getvar(ncfile, "Times", timeidx=frame+1).values)
        ax.set_title(f"NWP Hourly Rainfall Accumulation\nValid: {current_time_eat}", fontsize=14, fontweight="bold")
        return new_plot

    # interval=1500 makes each frame last 1.5 seconds (slower)
    ani = animation.FuncAnimation(fig, update, frames=ntimes-1, interval=1500)
    # fps=1 makes the video play at 1 frame per second (slow and clear)
    ani.save(os.path.join(output_dir, "hourly_rainfall_animation.mp4"), writer=FFMpegWriter(fps=1), dpi=150)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    run_rainfall_pipeline(args.input, args.outdir)
