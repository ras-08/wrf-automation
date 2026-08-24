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
from cartopy.mpl.path import shapely_to_path
import matplotlib.image as mpimg

def add_weatherex_logo(fig):
    logo = mpimg.imread(
        "/home/ras_08/WEATHER/INDIA/WORKFLOW/logo/weatherex_logo.png"
    )

    logo_ax = fig.add_axes(
        [0.01, 0.80, 0.10, 0.10]
    )

    logo_ax.imshow(logo)
    logo_ax.axis("off")

def run_temperature_pipeline(wrf_path, output_dir):
    ncfile = Dataset(wrf_path)
    shapefile_path = "/home/ras_08/WEATHER/INDIA/WORKFLOW/map_json_india/india_found/Admin2.shp"
    counties = gpd.read_file(shapefile_path)
    states = counties.dissolve(by="ST_NM")
    label = str(states)
    
    india_border = counties.union_all() if hasattr(counties, "union_all") else counties.unary_union
    
    # --- COLD TEMPERATURE PALETTE (Blues, Purples, Cyans) ---
    # Shifted to look "colder" for minimum temperature ranges
    min_temp_colors = [
        "#08306b", "#08519c", "#2171b5", "#4292c6", # Deep blues for cold
        "#6baed6", "#9ecae1", "#c6dbef", "#deebf7", # Light blues
        "#f7fbff", "#e0f3f8", "#abd9e9",             # Near-white to cyan
        "#74add1", "#4575b4", "#313695"              # Indigo/Deep cold
    ]
    # Adjusted levels to capture cold mornings (4°C to 24°C)
    levels = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
    cmap = ListedColormap(min_temp_colors)
    norm = BoundaryNorm(levels, cmap.N)

    t2_raw = getvar(ncfile, "T2", timeidx=0)
    lats, lons = latlon_coords(t2_raw)
    lon_min, lon_max = to_np(lons).min(), to_np(lons).max()
    lat_min, lat_max = to_np(lats).min(), to_np(lats).max()

    def format_eat_time(wrf_time_raw):
        ts = pd.to_datetime(str(wrf_time_raw))
        return (ts + timedelta(hours=3)).strftime('%d %b %Y, %H:%M EAT')

    def apply_map_features(axis):
        axis.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        axis.add_feature(cfeature.OCEAN, facecolor="#0974b3", zorder=0)
        axis.add_feature(cfeature.LAKES, facecolor="#0974b3", edgecolor="blue", linewidth=0.5, zorder=10)
        axis.add_feature(cfeature.RIVERS, edgecolor="#0974b3", linewidth=0.8, zorder=10)
        axis.add_feature(cfeature.COASTLINE, linewidth=1.5, edgecolor="black", zorder=11)
        axis.add_feature(cfeature.BORDERS, edgecolor="black", linewidth=1.5, zorder=5)
        counties.boundary.plot(ax=axis, edgecolor="black", linewidth=0.2, zorder=12)
        
        # State boundaries
        states.boundary.plot(ax=axis,edgecolor="black",linewidth=1.2,zorder=5)
        for state_name, row in states.iterrows():
            point = row.geometry.representative_point()

            axis.text(
                point.x,
                point.y,
                state_name,
                transform=ccrs.PlateCarree(),
                fontsize=6,
                color="blue",
                fontweight="bold",
                ha="center",
                va="center",
                zorder=10
                )

        # for idx, row in counties.iterrows():
        #     if 'NAME_LANG1' in row and pd.notnull(row['NAME_LANG1']):
        #         axis.text(
        #             row.geometry.centroid.x, 
        #             row.geometry.centroid.y, 
        #             row['NAME_LANG1'], 
        #             transform=ccrs.PlateCarree(), 
        #             fontsize=5, 
        #             fontstyle='italic',
        #             fontweight='bold', 
        #             ha='center', 
        #             zorder=15)

        # style = {'transform': ccrs.PlateCarree(), 'fontsize': 10, 'fontweight': 'bold', 'ha': 'center', 'zorder': 13}
        # axis.text(38.0, 0.5, "KENYA", **style)
        # axis.text(42.0, 2.0, "SOMALIA", **style)
        # axis.text(39.0, 5.0, "ETHIOPIA", **style)
        # axis.text(34.0, 2.0, "UGANDA", **style)
        # axis.text(35.0, -3.0, "TANZANIA", **style)
        # axis.text(34.0, 4.8, "SOUTH SUDAN", **style)
        # axis.text(41.5, -3.0, "INDIAN OCEAN", **style)

        gl = axis.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.3, linestyle='--', zorder=14)
        gl.top_labels = gl.right_labels = False
        gl.xformatter, gl.yformatter = LONGITUDE_FORMATTER, LATITUDE_FORMATTER
        gl.xlabel_style = {'fontsize': 10, 'color': 'black'}
        gl.ylabel_style = {'fontsize': 10, 'color': 'black'}

    paths = shapely_to_path(india_border)
    india_path = paths[0] if isinstance(paths, list) else paths

    # --- MINIMUM TEMPERATURE PLOT ---
    print("Generating Min Temp map...")
    min_t2 = np.min(getvar(ncfile, "T2", timeidx=None) - 273.15, axis=0)

    fig = plt.figure(figsize=(12, 14))
    add_weatherex_logo(fig)
    ax = plt.axes(projection=ccrs.PlateCarree())
    apply_map_features(ax)
    
    contour = ax.contourf(to_np(lons), to_np(lats), to_np(min_t2),
                          levels=levels, cmap=cmap, norm=norm, extend="both", 
                          transform=ccrs.PlateCarree(), zorder=1)
    
    for artist in ax.get_children():
        if getattr(artist, "__class__", "").__name__ in ['GeoContourSet', 'ContourSet']:
            if hasattr(artist, 'collections'):
                for col in artist.collections:
                    col.set_clip_path(india_path, transform=ax.transData)
            else:
                artist.set_clip_path(india_path, transform=ax.transData)

    plt.colorbar(contour, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7, label="Min Temperature (°C)")
    start_time_eat = format_eat_time(getvar(ncfile, "Times", timeidx=0).values)
    end_time_eat = format_eat_time(getvar(ncfile, "Times", timeidx=-1).values)
    plt.title(f"Minimum Temperature Forecast\nValid: {start_time_eat} TO: {end_time_eat}", 
              fontsize=14, fontweight="bold")
    
    plt.savefig(os.path.join(output_dir, "minimum_temperature.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # --- HOURLY ANIMATION ---
    print("Generating Minimum Temperature Animation...")
    fig, ax = plt.subplots(figsize=(12, 14), subplot_kw={'projection': ccrs.PlateCarree()})
    add_weatherex_logo(fig)
    apply_map_features(ax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7, label="Minimum Temperature (°C)")
    
    def update(frame):
        for c in list(ax.collections) + list(ax.artists):
            if getattr(c, 'zorder', 0) == 1:
                c.remove()
        
        t2_c = getvar(ncfile, "T2", timeidx=frame) - 273.15
        new_plot = ax.contourf(to_np(lons), to_np(lats), to_np(t2_c),
                               levels=levels, cmap=cmap, norm=norm, extend="both", 
                               transform=ccrs.PlateCarree(), zorder=1)
        
        if hasattr(new_plot, 'collections'):
            for col in new_plot.collections:
                col.set_clip_path(india_path, transform=ax.transData)
        else:
            new_plot.set_clip_path(india_path, transform=ax.transData)

        # Fixed time indexing for the title
        ax.set_title(f"Hourly Minimum Temperature Evolution\nValid: {format_eat_time(getvar(ncfile, 'Times', timeidx=(frame-1)).values)} to {format_eat_time(getvar(ncfile, 'Times', timeidx=frame).values)}",
                     fontsize=14, fontweight="bold")
        return [new_plot]

    ani = animation.FuncAnimation(fig, update, frames=ncfile.dimensions['Time'].size, interval=1200)
    ani.save(os.path.join(output_dir, "hourly_min_temp_animation.mp4"), writer=FFMpegWriter(fps=1), dpi=150)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    run_temperature_pipeline(args.input, args.outdir)
