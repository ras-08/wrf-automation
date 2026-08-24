import os
import argparse
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.animation import FFMpegWriter
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from netCDF4 import Dataset
from wrf import getvar, latlon_coords, to_np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.image as mpimg

# =============================================================================
#  CONFIGURATION
# =============================================================================

SHAPEFILE_PATH = "/home/ras_08/WEATHER/INDIA/WORKFLOW/map_json_india/india_found/Admin2.shp"

COLORS = [
    "#ffff99",  # 1   – 2.5  mm
    "#CECE0C",  # 2.5 – 10   mm
    "#0BFF0B",  # 10  – 20   mm
    "#1B5E20",  # 20  – 40   mm
    "#00B7FF",  # 40  – 70   mm
    "#0A2683",  # 70  – 130  mm
    "#F3931D",  # 130 – 200  mm
    "#FF0000",  # 200 – 300  mm
]

LEVELS = [1, 2.5, 10, 20, 40, 70, 130, 200, 300]


# =============================================================================
# META DATA
# =============================================================================

NWP_RES = "9 Km"
MODEL_NAME = "WFS (WeatherEx Forecasting System)"
DOMAIN_NAME = "INDIA"


# =============================================================================
#  HELPER FUNCTIONS
# =============================================================================

def load_geodata():
    counties = gpd.read_file(SHAPEFILE_PATH)
    states   = counties.dissolve(by="ST_NM")
    return counties, states


def build_colormap():
    cmap = ListedColormap(COLORS)
    norm = BoundaryNorm(LEVELS, cmap.N)
    return cmap, norm


def fmt_time(ts):
    return ts.strftime("%H UTC %d-%b-%Y")

def apply_map_features(axis, counties, states, lon_min, lon_max, lat_min, lat_max):
    axis.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    axis.add_feature(cfeature.LAND,      facecolor="white", zorder=0)
    axis.add_feature(cfeature.COASTLINE, linewidth=1.5,     zorder=4)
    counties.boundary.plot(ax=axis, edgecolor="#FF00FF", linewidth=0.2, zorder=4)
    states.boundary.plot(ax=axis,  edgecolor="#FF00FF", linewidth=1.2, zorder=5)

    for state_name, row in states.iterrows():
        point = row.geometry.representative_point()
        axis.text(
            point.x, point.y, state_name,
            transform=ccrs.PlateCarree(),
            fontsize=5.5, color="#FF00FF", fontweight="bold",
            ha="center", va="center", zorder=10,
            path_effects=[pe.withStroke(linewidth=1.5, foreground="white")]
        )

    gl = axis.gridlines(
        crs=ccrs.PlateCarree(), draw_labels=True,
        linewidth=0.8, color="gray", alpha=0.4, linestyle="--"
    )
    gl.top_labels   = False
    gl.right_labels = False
    gl.xformatter   = LONGITUDE_FORMATTER
    gl.yformatter   = LATITUDE_FORMATTER
    gl.xlabel_style = {"fontsize": 9, "color": "black"}
    gl.ylabel_style = {"fontsize": 9, "color": "black"}
def add_weatherex_logo(fig):
    logo = mpimg.imread(
        "/home/ras_08/WEATHER/INDIA/WORKFLOW/logo/weatherex_logo.png"
    )

    logo_ax = fig.add_axes(
        [0.02, 0.78, 0.18, 0.18]
    )

    logo_ax.imshow(logo)
    logo_ax.axis("off")

def add_bfs_colorbar(fig, ax, mappable):

    cbar = fig.colorbar(
        mappable,
        ax=ax,
        orientation="vertical",
        pad=0.02,
        shrink=0.5,
        ticks=LEVELS,
        extend="max"
    )

    cbar.ax.tick_params(labelsize=9)

    return cbar

def set_title(ax, param_label, forecast_hours, based_on_ts, valid_start_ts):

    valid_end = valid_start_ts + timedelta(hours=forecast_hours)

    based_on_str = based_on_ts.strftime("%H UTC %d-%b-%Y")

    valid_str = (
        f"{valid_start_ts.strftime('%H UTC %d-%b-%Y')} "
        f"to {valid_end.strftime('%H UTC %d-%b-%Y')}"
    )

    ax.set_title(
        f"{MODEL_NAME} ({NWP_RES}) {DOMAIN_NAME} {param_label} ({forecast_hours}-HR FCST)\n"
        f"Based on: {based_on_str}    Valid for: {valid_str}",
        fontsize=10,
        fontweight="bold",
        family="monospace",
        pad=8,
        linespacing=1.6,
    )


def add_footer(fig):

    fig.text(
        0.5,
        0.005,
        "(Background does not depict political boundary)",
        ha="center",
        fontsize=7.5,
        style="italic",
        color="#555555",
    )

    fig.text(
        0.01,
        0.005,
        "NWP MODEL OUTPUT | WeatherEx.Ai",
        ha="left",
        fontsize=7,
        color="#555555",
    )


def add_imd_logo_text(fig):
    fig.text(0.01, 0.005,
             "",
             ha="left", fontsize=6.5, color="#555555")


def remove_contourf(cf):
    """Remove contourf safely on Matplotlib < 3.8 and >= 3.8."""
    if cf is None:
        return
    try:
        cf.remove()
    except AttributeError:
        for coll in cf.collections:
            coll.remove()


def make_single_plot(fig, ax, lons, lats, data, cmap, norm,
                     param_label, forecast_hours,
                     based_on_ts, valid_start_ts,
                     counties, states,
                     lon_min, lon_max, lat_min, lat_max):
    """Draw one rainfall contourf — reused for every plot type."""
    apply_map_features(ax, counties, states, lon_min, lon_max, lat_min, lat_max)
    cf = ax.contourf(
        to_np(lons), to_np(lats), data,
        levels=LEVELS, cmap=cmap, norm=norm,
        extend="max", transform=ccrs.PlateCarree(),
        zorder=2, alpha=0.9
    )
    add_bfs_colorbar(fig, ax, cf)
    set_title(
    ax,
    param_label=param_label,
    forecast_hours=forecast_hours,
    based_on_ts=based_on_ts,
    valid_start_ts=valid_start_ts
    )
    add_footer(fig)
    add_imd_logo_text(fig)
    return cf


def get_day_intervals(ncfile, ntimes):
    """
    Group WRF time indices into full-day buckets: 00 UTC → 00 UTC next day.

    Returns a list of dicts:
        {
            'start_idx' : int,   # time index for 00 UTC of day N
            'end_idx'   : int,   # time index for 00 UTC of day N+1
            'start_ts'  : pd.Timestamp,
            'end_ts'    : pd.Timestamp,
        }

    Logic:
        - Collect all timestamps and their indices.
        - Find every index whose hour == 0 (i.e. 00 UTC).
        - Pair consecutive 00 UTC indices as (start, end) of each day.
        - If the first time-step is not 00 UTC, start the first day from
          index 0 and close it at the next 00 UTC found.
        - If there is no closing 00 UTC for the last open day,
          close it at the final time index (partial day).
    """
    # Build full timestamp list
    all_ts = [
        pd.to_datetime(str(getvar(ncfile, "Times", timeidx=i).values))
        for i in range(ntimes)
    ]

    # Indices where the hour is exactly 00 UTC
    midnight_indices = [i for i, ts in enumerate(all_ts) if ts.hour == 0]

    intervals = []

    if not midnight_indices:
        # No midnight found — treat entire file as one "day"
        intervals.append({
            "start_idx": 0,
            "end_idx":   ntimes - 1,
            "start_ts":  all_ts[0],
            "end_ts":    all_ts[-1],
        })
        return intervals

    # If file doesn't start at midnight, add a partial first day
    if midnight_indices[0] != 0:
        intervals.append({
            "start_idx": 0,
            "end_idx":   midnight_indices[0],
            "start_ts":  all_ts[0],
            "end_ts":    all_ts[midnight_indices[0]],
        })

    # Pair consecutive midnight indices as full days
    for k in range(len(midnight_indices) - 1):
        s = midnight_indices[k]
        e = midnight_indices[k + 1]
        intervals.append({
            "start_idx": s,
            "end_idx":   e,
            "start_ts":  all_ts[s],
            "end_ts":    all_ts[e],
        })

    # If the last midnight is not the final time-step, add a trailing partial day
    last_mid = midnight_indices[-1]
    if last_mid < ntimes - 1:
        intervals.append({
            "start_idx": last_mid,
            "end_idx":   ntimes - 1,
            "start_ts":  all_ts[last_mid],
            "end_ts":    all_ts[-1],
        })

    return intervals


# =============================================================================
#  MAIN PIPELINE
# =============================================================================

def run_rainfall_pipeline(wrf_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # Sub-folder for per-day plots
    daily_dir = os.path.join(output_dir, "daily_rainfall")
    os.makedirs(daily_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. LOAD DATA
    # -------------------------------------------------------------------------
    print("[1/4] Loading WRF file and shapefiles...")
    ncfile           = Dataset(wrf_path)
    counties, states = load_geodata()
    cmap, norm       = build_colormap()

    rain_t0          = getvar(ncfile, "RAINC", timeidx=0) + getvar(ncfile, "RAINNC", timeidx=0)
    lats, lons       = latlon_coords(rain_t0)
    lon_min, lon_max = float(to_np(lons).min()), float(to_np(lons).max())
    lat_min, lat_max = float(to_np(lats).min()), float(to_np(lats).max())

    ntimes       = ncfile.dimensions["Time"].size
    start_ts    = pd.to_datetime(str(getvar(ncfile, "Times", timeidx=0).values))
    end_ts      = pd.to_datetime(str(getvar(ncfile, "Times", timeidx=-1).values))
    delta_hours = int((end_ts - start_ts).total_seconds() // 3600)

    # -------------------------------------------------------------------------
    # 2. FULL CUMULATIVE RAINFALL MAP
    # -------------------------------------------------------------------------
    print("[2/4] Generating full cumulative rainfall map...")

    rain_end   = getvar(ncfile, "RAINC", timeidx=-1) + getvar(ncfile, "RAINNC", timeidx=-1)
    rain_total = np.ma.masked_less(to_np(rain_end) - to_np(rain_t0), 1.0)

    fig, ax = plt.subplots(figsize=(8, 12), subplot_kw={"projection": ccrs.PlateCarree()})
    add_weatherex_logo(fig)
    make_single_plot(
    fig, ax, lons, lats, rain_total, cmap, norm,
    param_label="RAINFALL (mm)",
    forecast_hours=delta_hours,
    based_on_ts=start_ts,
    valid_start_ts=start_ts,
    counties=counties, states=states,
    lon_min=lon_min, lon_max=lon_max,
    lat_min=lat_min, lat_max=lat_max
    )
    out_cum = os.path.join(output_dir, "cumulative_rainfall.png")
    plt.savefig(out_cum, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out_cum}")

    # -------------------------------------------------------------------------
    # 3. PER-DAY ACCUMULATED PLOTS  (00 UTC → 00 UTC next day)
    # -------------------------------------------------------------------------
    day_intervals = get_day_intervals(ncfile, ntimes)
    print(f"[3/4] Generating {len(day_intervals)} daily accumulated plot(s)...")

    for day_num, interval in enumerate(day_intervals, start=1):
        s_idx    = interval["start_idx"]
        e_idx    = interval["end_idx"]
        s_ts     = interval["start_ts"]
        e_ts     = interval["end_ts"]
        day_hrs  = int((e_ts - s_ts).total_seconds() // 3600)

        # Rainfall accumulated between the two boundary time-steps
        r_start = to_np(getvar(ncfile, "RAINC",  timeidx=s_idx) +
                        getvar(ncfile, "RAINNC", timeidx=s_idx))
        r_end   = to_np(getvar(ncfile, "RAINC",  timeidx=e_idx) +
                        getvar(ncfile, "RAINNC", timeidx=e_idx))
        daily   = np.ma.masked_less(r_end - r_start, 1.0)

        fig, ax = plt.subplots(
            figsize=(12, 14),
            subplot_kw={"projection": ccrs.PlateCarree()}
        )
        add_weatherex_logo(fig)
        make_single_plot(
            fig, ax, lons, lats, daily, cmap, norm,
            param_label="RAINFALL (mm)",
            forecast_hours=day_hrs,
            based_on_ts=start_ts,
            valid_start_ts=s_ts,
            counties=counties, states=states,
            lon_min=lon_min, lon_max=lon_max,
            lat_min=lat_min, lat_max=lat_max
        )
        # e.g.  daily_rainfall_Day1_06Jun2026.png
        fname    = f"daily_rainfall_Day{day_num}_{s_ts.strftime('%d%b%Y')}.png"
        out_path = os.path.join(daily_dir, fname)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"    Day {day_num}: {s_ts.strftime('%d %b %Y %H UTC')} → "
              f"{e_ts.strftime('%d %b %Y %H UTC')}  →  {fname}")

    # -------------------------------------------------------------------------
    # 4. HOURLY ANIMATION
    # -------------------------------------------------------------------------
    print("[4/4] Generating hourly rainfall animation...")

    fig2, ax2 = plt.subplots(
        figsize=(12, 14),
        subplot_kw={"projection": ccrs.PlateCarree()}
    )
    add_weatherex_logo(fig2)
    apply_map_features(ax2, counties, states, lon_min, lon_max, lat_min, lat_max)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    add_bfs_colorbar(fig2, ax2, sm)
    add_footer(fig2)
    add_imd_logo_text(fig2)

    _current_cf = [None]

    def update(frame):
        remove_contourf(_current_cf[0])

        r_prev = to_np(getvar(ncfile, "RAINC",  timeidx=frame)   +
                       getvar(ncfile, "RAINNC", timeidx=frame))
        r_now  = to_np(getvar(ncfile, "RAINC",  timeidx=frame+1) +
                       getvar(ncfile, "RAINNC", timeidx=frame+1))
        hourly = np.ma.masked_less_equal(r_now - r_prev, 0.1)

        cf_new = ax2.contourf(
            to_np(lons), to_np(lats), hourly,
            levels=LEVELS, cmap=cmap, norm=norm,
            extend="max", transform=ccrs.PlateCarree(),
            zorder=2, alpha=0.85
        )
        _current_cf[0] = cf_new

        t_prev = pd.to_datetime(str(getvar(ncfile, "Times", timeidx=frame).values))
        t_now  = pd.to_datetime(str(getvar(ncfile, "Times", timeidx=frame+1).values))
        set_title(
           ax2,
           param_label="RAINFALL (mm)",
           forecast_hours=1,
           based_on_ts=start_ts,
           valid_start_ts=t_prev,
        )
        return []

    ani = animation.FuncAnimation(
        fig2, update, frames=ntimes - 1,
        interval=1500, blit=False
    )
    out_ani = os.path.join(output_dir, "hourly_rainfall_animation.mp4")
    ani.save(out_ani, writer=FFMpegWriter(fps=1), dpi=150)
    plt.close()
    print(f"    Saved: {out_ani}")

    ncfile.close()
    print("\nDone! All outputs saved to:", output_dir)
    print(f"  • cumulative_rainfall.png          — full period total")
    print(f"  • hourly_rainfall_animation.mp4    — hourly animation")
    print(f"  • daily_rainfall/                  — {len(day_intervals)} daily PNG(s)")


# =============================================================================
#  ENTRY POINT
# =============================================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="NWP India Rainfall output"
    )

    today = datetime.now().strftime("%Y-%m-%d")

    DEFAULT_INPUT = (
        f"/home/ras_08/Models/WRF_TUTORIAL/WRFV4.5/run/"
        f"wrfout_d01_{today}_00:00:00"
    )

    DEFAULT_OUTDIR = (
        f"/home/ras_08/WEATHER/INDIA/Data/"
        f"{datetime.now().strftime('%Y%m%d')}"
    )

    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="WRF output file"
    )

    parser.add_argument(
        "--outdir",
        default=DEFAULT_OUTDIR,
        help="Output directory"
    )

    args = parser.parse_args()

    run_rainfall_pipeline(
        args.input,
        args.outdir
    )

