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
        [0.02, 0.80, 0.15, 0.15]
    )

    logo_ax.imshow(logo)
    logo_ax.axis("off")

def run_temperature_pipeline(wrf_path, output_dir):
    ncfile = Dataset(wrf_path)
    shapefile_path = "/home/ras_08/WEATHER/INDIA/WORKFLOW/map_json_india/india_found/Admin2.shp"
    counties = gpd.read_file(shapefile_path)
    states = counties.dissolve(by="ST_NM")
    label = str(states)
    
    # Handle union for different geopandas versions
    india_border = counties.union_all() if hasattr(counties, "union_all") else counties.unary_union
    
    # --- KMD Professional Palette ---
    kmd_colors = [
        "#4d004b", "#810f7c", "#88419d", "#8c6bb1", "#8c96c6", 
        "#9ebcda", "#bfd3e6", "#edf8fb",                      
        "#ffffcc", "#ffeda0", "#fed976",                      
        "#feb24c", "#fd8d3c", "#fc4e2a",                      
        "#e31a1c", "#bd0026", "#800026"                       
    ]
    levels = [10, 14, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 42]
    cmap = ListedColormap(kmd_colors)
    norm = BoundaryNorm(levels, cmap.N)

    # Coordinates
    t2_raw = getvar(ncfile, "T2", timeidx=0)
    lats, lons = latlon_coords(t2_raw)
    lon_min, lon_max = to_np(lons).min(), to_np(lons).max()
    lat_min, lat_max = to_np(lats).min(), to_np(lats).max()

    def format_eat_time(wrf_time_raw):
        ts = pd.to_datetime(str(wrf_time_raw))
        return (ts + timedelta(hours=0)).strftime('%d %b %Y, %H:%M UTC')

    # def apply_map_features(axis):
    #     axis.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    #     axis.add_feature(cfeature.OCEAN, facecolor="#1a5276", zorder=0)
    #     axis.add_feature(cfeature.LAKES, facecolor="#5dade2", edgecolor="#2e86c1", linewidth=0.5, zorder=10)
    #     axis.add_feature(cfeature.RIVERS, edgecolor="#5dade2", linewidth=0.8, zorder=10)
    #     axis.add_feature(cfeature.COASTLINE, linewidth=1.5, edgecolor="black", zorder=11)
    #     axis.add_feature(cfeature.BORDERS, edgecolor="black", linewidth=1.5, zorder=11)
        
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

        # # ---COUNTY LABELS ---
        # for idx, row in counties.iterrows():
        #     if 'NAME_LANG1' in row and pd.notnull(row['NAME_LANG1']):
        #         axis.text(
        #             row.geometry.centroid.x, 
        #             row.geometry.centroid.y, 
        #             row['NAME_LANG1'], 
        #             transform=ccrs.PlateCarree(), 
        #             fontsize=6, 
        #             fontstyle='italic',
        #             fontweight='bold', 
        #             ha='center',
        #             zorder=15 # High zorder to stay above the temperature data
        #         )

        # Labels for Neighboring Countries
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

    # Get the Path once
    paths = shapely_to_path(india_border)
    india_path = paths[0] if isinstance(paths, list) else paths

    # --- MAXIMUM TEMPERATURE PLOT ---
    print("Generating Max Temp map...")
    max_t2 = np.max(getvar(ncfile, "T2", timeidx=None) - 273.15, axis=0)

    fig = plt.figure(figsize=(12, 14))
    add_weatherex_logo(fig)
    ax = plt.axes(projection=ccrs.PlateCarree())
    # apply_map_features(ax)
    
    contour = ax.contourf(to_np(lons), to_np(lats), to_np(max_t2),
                          levels=levels, cmap=cmap, norm=norm, extend="both", 
                          transform=ccrs.PlateCarree(), zorder=1)
    
    # Clipping Fix for all versions
    for artist in ax.get_children():
        if isinstance(artist, plt.matplotlib.contour.ContourSet) or \
           getattr(artist, "__class__", "").__name__ == 'GeoContourSet':
            if hasattr(artist, 'collections'):
                for col in artist.collections:
                    col.set_clip_path(india_path, transform=ax.transData)
            else:
                artist.set_clip_path(india_path, transform=ax.transData)

    plt.colorbar(contour, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7, label="Maximum Temp (°C)")
    start_time_eat = format_eat_time(getvar(ncfile, "Times", timeidx=0).values)
    end_time_eat = format_eat_time(getvar(ncfile, "Times", timeidx=-1).values)
    plt.title(f"NWP Maximum Temperature Forecast\nValid: {start_time_eat} to {end_time_eat}", 
              fontsize=14, fontweight="bold")
    
    plt.savefig(os.path.join(output_dir, "maximum_temperature.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # --- HOURLY ANIMATION ---
    print("Generating Maximum Temperature Animation...")
    fig, ax = plt.subplots(figsize=(12, 14), subplot_kw={'projection': ccrs.PlateCarree()})
    
    add_weatherex_logo(fig)
    # apply_map_features(ax)

    # colorar for animation
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7, label="Hourly 2m Temperature (°C)")
    
    def update(frame):
        # Clear only previous contours
        for c in list(ax.collections) + list(ax.artists):
            if getattr(c, 'zorder', 0) == 1:
                c.remove()
        
        t2_c = getvar(ncfile, "T2", timeidx=frame) - 273.15
        new_plot = ax.contourf(to_np(lons), to_np(lats), to_np(t2_c),
                               levels=levels, cmap=cmap, norm=norm, extend="both", 
                               transform=ccrs.PlateCarree(), zorder=1)
        
        # Apply clipping to frame
        if hasattr(new_plot, 'collections'):
            for col in new_plot.collections:
                col.set_clip_path(india_path, transform=ax.transData)
        else:
            new_plot.set_clip_path(india_path, transform=ax.transData)

        ax.set_title(f"WRF Hourly Maximum Temperature Evolution\nValid: {format_eat_time(getvar(ncfile, 'Times', timeidx=(frame-1)).values)} to {format_eat_time(getvar(ncfile, 'Times', timeidx=frame).values)}", 
                     fontsize=14, fontweight="bold")
        return [new_plot]

    ani = animation.FuncAnimation(fig, update, frames=ncfile.dimensions['Time'].size, interval=1200)
    ani.save(os.path.join(output_dir, "hourly_max_temp_animation.mp4"), writer=FFMpegWriter(fps=1), dpi=150)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    run_temperature_pipeline(args.input, args.outdir)
