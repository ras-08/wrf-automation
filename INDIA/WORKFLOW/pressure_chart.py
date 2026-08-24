import os
import argparse
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from netCDF4 import Dataset
from wrf import getvar, latlon_coords, to_np
import pandas as pd


SHAPEFILE_PATH = "/home/ras_08/WEATHER/INDIA/WORKFLOW/map_json_india/india_found/Admin2.shp"

MODEL_NAME = "WFS (WeatherEx Forecasting System)"
NWP_RES = "9 Km"
DOMAIN_NAME = "INDIA"


def load_geodata():
    counties = gpd.read_file(SHAPEFILE_PATH)

    # Important: cartopy plotting expects lon/lat coordinates
    if counties.crs is not None and counties.crs.to_epsg() != 4326:
        counties = counties.to_crs(epsg=4326)

    states = counties.dissolve(by="ST_NM")
    return counties, states


def apply_map_features(axis, counties, states, lon_min, lon_max, lat_min, lat_max):
    axis.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    axis.add_feature(cfeature.LAND, facecolor="white", zorder=0)
    axis.add_feature(cfeature.COASTLINE, linewidth=1.2, zorder=4)

    counties.boundary.plot(
        ax=axis,
        edgecolor="#777777",
        linewidth=0.2,
        zorder=4,
        transform=ccrs.PlateCarree()
    )

    states.boundary.plot(
        ax=axis,
        edgecolor="black",
        linewidth=0.8,
        zorder=5,
        transform=ccrs.PlateCarree()
    )

    for state_name, row in states.iterrows():
        point = row.geometry.representative_point()
        axis.text(
            point.x,
            point.y,
            state_name,
            transform=ccrs.PlateCarree(),
            fontsize=5.5,
            color="black",
            fontweight="bold",
            ha="center",
            va="center",
            zorder=10,
            path_effects=[pe.withStroke(linewidth=1.5, foreground="white")]
        )

    gl = axis.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.8,
        color="gray",
        alpha=0.4,
        linestyle="--"
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {"fontsize": 9, "color": "black"}
    gl.ylabel_style = {"fontsize": 9, "color": "black"}


def make_pressure_plot(wrf_path, output_dir, timeidx=0):
    os.makedirs(output_dir, exist_ok=True)

    ncfile = Dataset(wrf_path)
    counties, states = load_geodata()

    slp = getvar(ncfile, "slp", timeidx=timeidx)
    lats, lons = latlon_coords(slp)

    slp_np = to_np(slp)

    lon_min, lon_max = float(to_np(lons).min()), float(to_np(lons).max())
    lat_min, lat_max = float(to_np(lats).min()), float(to_np(lats).max())

    valid_time = pd.to_datetime(str(getvar(ncfile, "Times", timeidx=timeidx).values))

    fig, ax = plt.subplots(
        figsize=(10, 12),
        subplot_kw={"projection": ccrs.PlateCarree()}
    )

    apply_map_features(ax, counties, states, lon_min, lon_max, lat_min, lat_max)

    # Filled pressure shading
    levels = np.arange(980, 1042, 2)
    cf = ax.contourf(
        to_np(lons),
        to_np(lats),
        slp_np,
        levels=levels,
        cmap="coolwarm",
        extend="both",
        transform=ccrs.PlateCarree(),
        zorder=2,
        alpha=0.85
    )

    # red isobars
    isobar_levels = np.arange(980, 1042, 4)
    cs = ax.contour(
        to_np(lons),
        to_np(lats),
        slp_np,
        levels=isobar_levels,
        colors="red",
        linewidths=0.7,
        transform=ccrs.PlateCarree(),
        zorder=6
    )

    ax.clabel(cs, inline=True, fontsize=8, fmt="%d")

    cbar = fig.colorbar(
        cf,
        ax=ax,
        orientation="vertical",
        pad=0.02,
        shrink=0.65
    )
    cbar.set_label("Mean Sea Level Pressure (hPa)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    ax.set_title(
        f"{MODEL_NAME} ({NWP_RES}) {DOMAIN_NAME} Mean Sea Level Pressure\n"
        f"Valid: {valid_time.strftime('%H UTC %d-%b-%Y')}",
        fontsize=11,
        fontweight="bold",
        family="monospace",
        pad=10
    )

    fig.text(
        0.5,
        0.01,
        "(Background does not depict political boundary)",
        ha="center",
        fontsize=7.5,
        style="italic",
        color="#555555"
    )

    out_path = os.path.join(output_dir, f"slp_india_{valid_time.strftime('%Y%m%d_%HUTC')}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    ncfile.close()

    print("Saved:", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WRF India pressure chart")

    parser.add_argument("--input", required=True, help="WRF output file")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--timeidx", type=int, default=0, help="WRF time index")

    args = parser.parse_args()

    make_pressure_plot(args.input, args.outdir, args.timeidx)
