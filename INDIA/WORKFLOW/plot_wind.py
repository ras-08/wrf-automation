import os
import argparse
import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from matplotlib.animation import FFMpegWriter
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from netCDF4 import Dataset
from wrf import getvar, latlon_coords, to_np
import pandas as pd
from datetime import timedelta
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


def run_wind_vector_pipeline(wrf_path, output_dir):
    ncfile = Dataset(wrf_path)
    shapefile_path = "/home/ras_08/WEATHER/INDIA/WORKFLOW/map_json_india/india_found/Admin2.shp"
    counties = gpd.read_file(shapefile_path)
    states = counties.dissolve(by="ST_NM")
    label = str(states)
    
    
    # --- 1. HIGH-CONTRAST JET PALETTE ---
    colors = ["#9400D3", "#4B0082", "#0000FF", "#00FFFF", "#00FF00", "#FFFF00", "#FF7F00", "#FF0000"]
    levels = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15]
    cmap = LinearSegmentedColormap.from_list("vibrant_jet", colors, N=len(levels)-1)
    norm = BoundaryNorm(levels, ncolors=cmap.N)

    # --- 2. DATA EXTRACTION (m/s) ---
    uv_met = getvar(ncfile, "uvmet10", timeidx=None, units="m s-1")
    u, v = uv_met[0, :], uv_met[1, :]
    speed = np.sqrt(u**2 + v**2)
    lats, lons = latlon_coords(u)
    lon_min, lon_max = to_np(lons).min(), to_np(lons).max()
    lat_min, lat_max = to_np(lats).min(), to_np(lats).max()

    def format_eat_time(wrf_time_raw):
        ts = pd.to_datetime(str(wrf_time_raw))
        eat_time = ts + timedelta(hours=3)
        return eat_time.strftime('%d %b %Y | %H:%M EAT')

    def apply_map_features(axis):
        axis.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        axis.add_feature(cfeature.OCEAN, facecolor="#0974b3", edgecolor="black")
        axis.add_feature(cfeature.COASTLINE, linewidth=2, edgecolor="black", zorder=5)
        axis.add_feature(cfeature.LAND, facecolor="white", zorder=0)
        axis.add_feature(cfeature.LAKES, facecolor="#0974b3", edgecolor="black", zorder=5)
        axis.add_feature(cfeature.BORDERS, edgecolor="black", linewidth=2, zorder=5)
        axis.add_feature(cfeature.RIVERS, facecolor="#0974b3", edgecolor="black")

        counties.boundary.plot(ax=axis, edgecolor="black", linewidth=0.3,zorder=6)
        
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
        #         if 'NAME_LANG1' in row and pd.notnull(row['NAME_LANG1']):
        #             axis.text(
        #                 row.geometry.centroid.x, 
        #                 row.geometry.centroid.y, 
        #                 row['NAME_LANG1'], 
        #                 transform=ccrs.PlateCarree(), 
        #                 fontsize=7, 
        #                 fontstyle='italic',
        #                 fontweight='bold',
        #                 ha='center',
        #                 zorder=4
        #             )


        # style = {
        #     "transform": ccrs.PlateCarree(),
        #     "fontsize": 14,
        #     "fontweight": "bold",
        #     "ha": "center",
        #     "zorder": 7
        # }

        # axis.text(38.0, 0.5, "KENYA", **style)
        # axis.text(42.0, 2.0, "SOMALIA", **style)
        # axis.text(39.0, 5.0, "ETHIOPIA", **style)
        # axis.text(34.0, 2.0, "UGANDA", **style)
        # axis.text(35.0, -3.0, "TANZANIA", **style)
        # axis.text(34.0, 4.8, "SOUTH SUDAN", **style)
        # axis.text(41.5, -3.0, "INDIAN OCEAN", **style)

        gl = axis.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, 
                            color='grey', alpha=0.3, linestyle='--')
        gl.top_labels = gl.right_labels = False
        gl.xformatter, gl.yformatter = LONGITUDE_FORMATTER, LATITUDE_FORMATTER
        gl.xlabel_style = gl.ylabel_style = {"fontsize": 10, "color": "black"}

    # --- 3. STATIC PICTURE (Summary) ---
    print("Generating 24-hour Summary of Wind Speed Map...")
    fig = plt.figure(figsize=(12, 15))
    ax = plt.axes(projection=ccrs.PlateCarree())
    apply_map_features(ax)
    add_weatherex_logo(fig)
    max_speed = np.max(speed, axis=0)
    cf = ax.contourf(to_np(lons), to_np(lats), to_np(max_speed), 
                     levels=levels, cmap=cmap, norm=norm, extend="both", 
                     transform=ccrs.PlateCarree(), zorder=3)
    
    # overlying the vector field
    skip = 4
    vec = ax.quiver(to_np(lons[::skip, ::skip]), to_np(lats[::skip, ::skip]), 
                    to_np(np.mean(u, axis=0)[::skip, ::skip]), to_np(np.mean(v, axis=0)[::skip, ::skip]), 
                    transform=ccrs.PlateCarree(), color='black', scale=180, width=0.002, zorder=10)

    plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.03, shrink=0.8, 
                 label='Max Wind Speed (m/s)')
    
    start_t = format_eat_time(uv_met.Time.values[0])
    end_t = format_eat_time(uv_met.Time.values[-1])
    plt.title(f"MAX. SURFACE WIND FORECAST\nValid: {start_t} <--> {end_t}", 
              fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(output_dir, "wind_vibrant_picture.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # --- 4. HOURLY ANIMATION ---
    print("Generating Hourly Vector Animation...")
    fig, ax = plt.subplots(figsize=(12, 13), subplot_kw={'projection': ccrs.PlateCarree()})
    add_weatherex_logo(fig)

    # Static Colorbar for Animation
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, ax=ax, orientation="horizontal", pad=0.03, shrink=0.8, label="Max Wind Speed (m/s)")
       
    def update(frame):
        ax.clear()
        apply_map_features(ax)
        
        curr_speed = to_np(speed[frame, :])
        curr_u, curr_v = to_np(u[frame, :]), to_np(v[frame, :])

        cf_anim = ax.contourf(to_np(lons), to_np(lats), curr_speed, 
                              levels=levels, cmap=cmap, 
                              norm=norm, transform=ccrs.PlateCarree(), zorder=3)
        
        ax.quiver(to_np(lons[::skip, ::skip]), to_np(lats[::skip, ::skip]), 
                  curr_u[::skip, ::skip], curr_v[::skip, ::skip], 
                  transform=ccrs.PlateCarree(), color='black', scale=180, width=0.002, zorder=10)

        t_str = format_eat_time(uv_met.Time.values[frame])
        ax.set_title(f"HOURLY WIND EVOLUTION: {t_str}", fontsize=15, fontweight='bold')

    ani = animation.FuncAnimation(fig, update, frames=range(len(uv_met.Time)), interval=1500)
    ani.save(os.path.join(output_dir, "wind_evolution_vibrant.mp4"), writer=FFMpegWriter(fps=1), dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    run_wind_vector_pipeline(args.input, args.outdir)
