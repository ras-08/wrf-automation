import os
import argparse
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import PathPatch
from matplotlib.path import Path
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.io.shapereader as shpreader
from shapely.geometry import box
from shapely.ops import unary_union
from netCDF4 import Dataset
from wrf import getvar, latlon_coords, to_np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.image as mpimg


# =============================================================================
#  CONFIGURATION
# =============================================================================

SHAPEFILE_PATH = "/home/ras_08/WEATHER/INDIA/WORKFLOW/kerala/shapefiles/district.shp"

# Kerala bounding box (with small buffer)
KERALA_LON_MIN =  74.5
KERALA_LON_MAX =  78.0
KERALA_LAT_MIN =   7.9
KERALA_LAT_MAX =  12.8

COLORS = [
    "#FFFFFF",  # No Rain                 : 0
    "#00FF00",  # Light Rainfall          : 0.1   - 15.5  mm
    "#008000",  # Moderate Rainfall       : 15.6  - 64.4  mm
    "#FFFF00",  # Heavy Rainfall          : 64.5  - 115.5 mm
    "#FFA500",  # Very Heavy Rainfall     : 115.6 - 204.4 mm
    "#FF0000",  # Extremely Heavy Rainfall: > 204.4 mm
]

LEVELS = [0, 0.1, 15.6, 64.5, 115.6, 204.5, 300]

LEVEL_LABELS = [
    "No Rain",
    "Light\n(0.1-15.5)",
    "Moderate\n(15.6-64.4)",
    "Heavy\n(64.5-115.5)",
    "Very Heavy\n(115.6-204.4)",
    "Extremely Heavy\n(>204.4)",
]
# =============================================================================
# META DATA
# =============================================================================

NWP_RES = "9 Km"
MODEL_NAME = "WFS (WeatherEx Forecasting System)"
DOMAIN_NAME = "KERALA"


# =============================================================================
#  HELPER FUNCTIONS
# =============================================================================

def load_geodata():
    """Load shapefile and extract Kerala districts + boundary."""
    counties = gpd.read_file(SHAPEFILE_PATH)
    states   = counties.dissolve(by="ST_NM")

    # Kerala only
    kerala_districts = counties[counties["ST_NM"] == "Kerala"].copy()
    kerala_state     = states[states.index == "Kerala"].copy()

    # Reproject to PlateCarree-compatible CRS if needed
    if kerala_districts.crs is None or kerala_districts.crs.to_epsg() != 4326:
        kerala_districts = kerala_districts.set_crs(epsg=4326, allow_override=True)
        kerala_state     = kerala_state.set_crs(epsg=4326, allow_override=True)

    return counties, states, kerala_districts, kerala_state


def build_colormap():
    cmap = ListedColormap(COLORS)
    norm = BoundaryNorm(LEVELS, cmap.N)
    return cmap, norm

def fmt_time(ts):
    return ts.strftime("%H UTC %d-%b-%Y")

def add_kerala_mask(ax, kerala_state):
    """
    Add a white mask OUTSIDE Kerala so only Kerala data is visible.
    Uses a large bounding box minus Kerala polygon = the 'outside' region.
    """
    # Outer bounding box much larger than Kerala
    outer = box(60.0, -10.0, 100.0, 40.0)

    # Union of Kerala geometry
    kerala_union = unary_union(kerala_state.geometry.values)

    # Difference = everything outside Kerala
    outside = outer.difference(kerala_union)

    # Add as a white filled patch on top of the data
    ax.add_geometries(
        [outside],
        crs=ccrs.PlateCarree(),
        facecolor="white",
        edgecolor="none",
        zorder=3        # above contourf (zorder=2), below boundaries (zorder=4+)
    )


def apply_kerala_map_features(axis, kerala_districts, kerala_state):
    """Map decorations zoomed to Kerala with exact boundary masking."""

    axis.set_extent(
        [KERALA_LON_MIN, KERALA_LON_MAX, KERALA_LAT_MIN, KERALA_LAT_MAX],
        crs=ccrs.PlateCarree()
    )

    # Base: white land, coastline
    axis.add_feature(cfeature.LAND,      facecolor="white", zorder=0)
    axis.add_feature(cfeature.OCEAN,     facecolor="#cce6ff", zorder=0)
    axis.add_feature(cfeature.COASTLINE, linewidth=1.5,     zorder=5)

    # White mask outside Kerala (so data only shows inside Kerala)
    add_kerala_mask(axis, kerala_state)

    # District boundaries inside Kerala (thin magenta)
    kerala_districts.boundary.plot(
        ax=axis, edgecolor="#FF00FF", linewidth=0.5, zorder=6
    )

    # Kerala state outer boundary (thick magenta)
    kerala_state.boundary.plot(
        ax=axis, edgecolor="#FF00FF", linewidth=2.0, zorder=7
    )

    # District name labels
    for _, row in kerala_districts.iterrows():
        point = row.geometry.representative_point()
        dist_name = row.get("DISTRICT", row.get("dtname", row.get("NAME_2", "")))
        if dist_name:
            axis.text(
                point.x, point.y,
                dist_name,
                transform=ccrs.PlateCarree(),
                fontsize=5.5, color="#880088", fontweight="bold",
                ha="center", va="center", zorder=10,
                path_effects=[pe.withStroke(linewidth=1.5, foreground="white")]
            )

    # "KERALA" label at state centre
    k_point = kerala_state.geometry.iloc[0].representative_point()
    axis.text(
        k_point.x, k_point.y + 1.5,
        "KERALA",
        transform=ccrs.PlateCarree(),
        fontsize=11, color="#CC0000", fontweight="bold",
        ha="center", va="center", zorder=10,
        path_effects=[pe.withStroke(linewidth=2.5, foreground="white")]
    )

    # Gridlines
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
        [0.01, 0.80, 0.15, 0.15]   # left, bottom, width, height
    )

    logo_ax.imshow(logo)
    logo_ax.axis("off")


def add_bfs_colorbar(fig, ax, mappable):
    cbar = fig.colorbar(
        mappable, ax=ax, orientation="vertical",
        pad=0.02, shrink=0.50, fraction=0.03, ticks=[]
    )
    # Place one category label centered in each color band
    n_bands = len(COLORS)
    midpoints = [(LEVELS[i] + LEVELS[i + 1]) / 2 for i in range(n_bands)]
    cbar.set_ticks(midpoints)
    cbar.ax.set_yticklabels(LEVEL_LABELS, fontsize=7.5)
    cbar.ax.tick_params(length=0)
    return cbar



def set_title(ax, param_label, forecast_hours, based_on_ts, valid_start_ts=None, valid_end_ts=None):

    based_on_str = based_on_ts.strftime("%H UTC %d-%b-%Y")

    if valid_start_ts is not None and valid_end_ts is not None:
        valid_str = (
            f"{valid_start_ts.strftime('%H UTC %d-%b-%Y')} "
            f"to {valid_end_ts.strftime('%H UTC %d-%b-%Y')}"
        )
    else:
        valid_end = based_on_ts + timedelta(hours=forecast_hours)
        valid_str = (
            f"{based_on_ts.strftime('%H UTC %d-%b-%Y')} "
            f"to {valid_end.strftime('%H UTC %d-%b-%Y')}"
        )

    ax.set_title(
        f"{MODEL_NAME} ({NWP_RES}) {DOMAIN_NAME} {param_label} ({forecast_hours}-HR FCST)\n"
        f"Based on: {based_on_str}    Valid for: {valid_str}",
        fontsize=10, fontweight="bold",
        family="monospace", pad=8, linespacing=1.6,
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
    if cf is None:
        return
    try:
        cf.remove()
    except AttributeError:
        for coll in cf.collections:
            coll.remove()


def make_kerala_plot(fig, ax, lons, lats, data, cmap, norm,
                     param_label, forecast_hours,
                     based_on_ts, valid_start_ts, valid_end_ts,
                     kerala_districts, kerala_state):
    """Draw one Kerala-masked rainfall plot."""
    apply_kerala_map_features(ax, kerala_districts, kerala_state)

    '''cf = ax.pcolormesh(
        to_np(lons), to_np(lats), data,
        levels=LEVELS, cmap=cmap, norm=norm,
        extend="max", transform=ccrs.PlateCarree(),
        zorder=2, alpha=0.9
    )'''
    cf = ax.pcolormesh(
    to_np(lons),
    to_np(lats),
    data,
    cmap=cmap,
    norm=norm,
    shading="auto",
    transform=ccrs.PlateCarree(),
    zorder=2,
    alpha=0.9
    )
    add_bfs_colorbar(fig, ax, cf)
    set_title(ax,
        param_label=param_label,
        forecast_hours=forecast_hours,
        based_on_ts=based_on_ts,
        valid_start_ts=valid_start_ts,
        valid_end_ts=valid_end_ts,
    )
    add_footer(fig)
    add_imd_logo_text(fig)
    return cf


def get_day_intervals(ncfile, ntimes):
    """Group WRF time indices into 00 UTC → 00 UTC next day buckets."""
    all_ts = [
        pd.to_datetime(str(getvar(ncfile, "Times", timeidx=i).values))
        for i in range(ntimes)
    ]

    midnight_indices = [i for i, ts in enumerate(all_ts) if ts.hour == 0]
    intervals = []

    if not midnight_indices:
        intervals.append({
            "start_idx": 0, "end_idx": ntimes - 1,
            "start_ts": all_ts[0], "end_ts": all_ts[-1],
        })
        return intervals

    if midnight_indices[0] != 0:
        intervals.append({
            "start_idx": 0, "end_idx": midnight_indices[0],
            "start_ts": all_ts[0], "end_ts": all_ts[midnight_indices[0]],
        })

    for k in range(len(midnight_indices) - 1):
        s, e = midnight_indices[k], midnight_indices[k + 1]
        intervals.append({
            "start_idx": s, "end_idx": e,
            "start_ts": all_ts[s], "end_ts": all_ts[e],
        })

    last_mid = midnight_indices[-1]
    if last_mid < ntimes - 1:
        intervals.append({
            "start_idx": last_mid, "end_idx": ntimes - 1,
            "start_ts": all_ts[last_mid], "end_ts": all_ts[-1],
        })

    return intervals


# =============================================================================
#  MAIN PIPELINE
# =============================================================================

def run_kerala_rainfall_pipeline(wrf_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    daily_dir = os.path.join(output_dir, "daily_rainfall_kerala")
    os.makedirs(daily_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. LOAD DATA
    # -------------------------------------------------------------------------
    print("[1/4] Loading WRF file and shapefiles...")
    ncfile = Dataset(wrf_path)
    counties, states, kerala_districts, kerala_state = load_geodata()
    cmap, norm = build_colormap()

    rain_t0          = getvar(ncfile, "RAINC", timeidx=0) + getvar(ncfile, "RAINNC", timeidx=0)
    lats, lons       = latlon_coords(rain_t0)
    ntimes       = ncfile.dimensions["Time"].size
    start_ts    = pd.to_datetime(str(getvar(ncfile, "Times", timeidx=0).values))
    end_ts      = pd.to_datetime(str(getvar(ncfile, "Times", timeidx=-1).values))
    delta_hours = int((end_ts - start_ts).total_seconds() // 3600)

    # -------------------------------------------------------------------------
    # 2. FULL CUMULATIVE — KERALA
    # -------------------------------------------------------------------------
    print("[2/4] Generating Kerala cumulative rainfall map...")

    rain_end   = getvar(ncfile, "RAINC", timeidx=-1) + getvar(ncfile, "RAINNC", timeidx=-1)
    rain_total = np.ma.masked_less(to_np(rain_end) - to_np(rain_t0), 1.0)

    fig, ax = plt.subplots(figsize=(8, 14), subplot_kw={"projection": ccrs.PlateCarree()})
    make_kerala_plot(
        fig, ax, lons, lats, rain_total, cmap, norm,
        param_label="RAINFALL (mm)",
        forecast_hours=delta_hours,
        based_on_ts=start_ts,
        valid_start_ts=start_ts,
        valid_end_ts=end_ts,
        kerala_districts=kerala_districts,
        kerala_state=kerala_state
    )
    add_weatherex_logo(fig)
    
    out_cum = os.path.join(output_dir, "kerala_cumulative_rainfall.png")
    plt.savefig(out_cum, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out_cum}")

    # -------------------------------------------------------------------------
    # 3. PER-DAY PLOTS — KERALA  (00 UTC → 00 UTC next day)
    # -------------------------------------------------------------------------
    day_intervals = get_day_intervals(ncfile, ntimes)
    print(f"[3/4] Generating {len(day_intervals)} daily Kerala plot(s)...")

    for day_num, interval in enumerate(day_intervals, start=1):
        s_idx   = interval["start_idx"]
        e_idx   = interval["end_idx"]
        s_ts    = interval["start_ts"]
        e_ts    = interval["end_ts"]
        day_hrs = int((e_ts - s_ts).total_seconds() // 3600)

        r_start = to_np(getvar(ncfile, "RAINC",  timeidx=s_idx) +
                        getvar(ncfile, "RAINNC", timeidx=s_idx))
        r_end   = to_np(getvar(ncfile, "RAINC",  timeidx=e_idx) +
                        getvar(ncfile, "RAINNC", timeidx=e_idx))
        daily   = np.ma.masked_less(r_end - r_start, 1.0)

        fig, ax = plt.subplots(
            figsize=(8, 14),
            subplot_kw={"projection": ccrs.PlateCarree()}
        )
        make_kerala_plot(
            fig, ax, lons, lats, daily, cmap, norm,
            param_label="RAINFALL (mm)",
            forecast_hours=day_hrs,
            based_on_ts=start_ts,
            valid_start_ts=s_ts,
            valid_end_ts=e_ts,
            kerala_districts=kerala_districts,
            kerala_state=kerala_state
        )
        add_weatherex_logo(fig)
    
      

        fname    = f"kerala_daily_Day{day_num}_{s_ts.strftime('%d%b%Y')}.png"
        out_path = os.path.join(daily_dir, fname)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"    Day {day_num}: {s_ts.strftime('%d %b %Y %H UTC')} → "
              f"{e_ts.strftime('%d %b %Y %H UTC')}  →  {fname}")

    # -------------------------------------------------------------------------
    # 4. HOURLY ANIMATION — KERALA
    # -------------------------------------------------------------------------
    print("[4/4] Generating Kerala hourly animation...")

    fig2, ax2 = plt.subplots(
        figsize=(8, 14),
        subplot_kw={"projection": ccrs.PlateCarree()}
    )
    apply_kerala_map_features(ax2, kerala_districts, kerala_state)
    add_weatherex_logo(fig2)

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

        '''cf_new = ax2.contourf(
            to_np(lons), to_np(lats), hourly,
            levels=LEVELS, cmap=cmap, norm=norm,
            extend="max", transform=ccrs.PlateCarree(),
            zorder=2, alpha=0.85
        )'''
        cf_new = ax2.pcolormesh(
        to_np(lons),
        to_np(lats),
        hourly,
        cmap=cmap,
        norm=norm,
        shading="auto",
        transform=ccrs.PlateCarree(),
        zorder=2,
        alpha=0.85
        )
        _current_cf[0] = cf_new

        t_prev = pd.to_datetime(str(getvar(ncfile, "Times", timeidx=frame).values))
        t_now  = pd.to_datetime(str(getvar(ncfile, "Times", timeidx=frame+1).values))
        set_title(ax2,
            param_label="RAINFALL (mm)",
            forecast_hours=1,
            based_on_ts=start_ts,
            valid_start_ts=t_prev,
            valid_end_ts=t_now,
        )
        return []

    ani = animation.FuncAnimation(
        fig2, update, frames=ntimes - 1,
        interval=1500, blit=False
    )
    out_ani = os.path.join(output_dir, "kerala_hourly_animation.mp4")
    ani.save(out_ani, writer=FFMpegWriter(fps=1), dpi=150)
    plt.close()
    print(f"    Saved: {out_ani}")

    ncfile.close()
    print("\nDone! All Kerala outputs saved to:", output_dir)
    print(f"  • kerala_cumulative_rainfall.png")
    print(f"  • kerala_hourly_animation.mp4")
    print(f"  • daily_rainfall_kerala/  ({len(day_intervals)} daily PNG files)")


# =============================================================================
#  ENTRY POINT
# =============================================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="NWP Kerala Rainfall output"
    )

    today = datetime.now().strftime("%Y-%m-%d")

    DEFAULT_INPUT = (
        f"/home/ras_08/Models/WRF_TUTORIAL/WRFV4.5/run/"
        f"wrfout_d01_{today}_00:00:00"
    )

    DEFAULT_OUTDIR = (
        f"/home/ras_08/WEATHER/INDIA/Data/"
        f"{datetime.now().strftime('%Y%m%d')}/00z"
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

    run_kerala_rainfall_pipeline(
        args.input,
        args.outdir
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
             "IMD OPERATIONAL GLOBAL MODEL COURTESY : BTM, NCMRWF",
             ha="left", fontsize=6.5, color="#555555")


def remove_contourf(cf):
    if cf is None:
        return
    try:
        cf.remove()
    except AttributeError:
        for coll in cf.collections:
            coll.remove()


def make_kerala_plot(fig, ax, lons, lats, data, cmap, norm,
                     param_label, forecast_hours,
                     #based_on_str, valid_for_str,
                     based_on_ts,
                     kerala_districts, kerala_state):
    """Draw one Kerala-masked rainfall plot."""
    apply_kerala_map_features(ax, kerala_districts, kerala_state)

    '''cf = ax.pcolormesh(
        to_np(lons), to_np(lats), data,
        levels=LEVELS, cmap=cmap, norm=norm,
        extend="max", transform=ccrs.PlateCarree(),
        zorder=2, alpha=0.9
    )'''
    cf = ax.pcolormesh(
    to_np(lons),
    to_np(lats),
    data,
    cmap=cmap,
    norm=norm,
    shading="auto",
    transform=ccrs.PlateCarree(),
    zorder=2,
    alpha=0.9
    )
    add_bfs_colorbar(fig, ax, cf)
    set_title(ax,
        param_label=param_label,
        forecast_hours=forecast_hours,
        based_on_ts=based_on_ts,
        #valid_for_str=valid_for_str
    )
    add_footer(fig)
    add_imd_logo_text(fig)
    return cf


def get_day_intervals(ncfile, ntimes):
    """Group WRF time indices into 00 UTC → 00 UTC next day buckets."""
    all_ts = [
        pd.to_datetime(str(getvar(ncfile, "Times", timeidx=i).values))
        for i in range(ntimes)
    ]

    midnight_indices = [i for i, ts in enumerate(all_ts) if ts.hour == 0]
    intervals = []

    if not midnight_indices:
        intervals.append({
            "start_idx": 0, "end_idx": ntimes - 1,
            "start_ts": all_ts[0], "end_ts": all_ts[-1],
        })
        return intervals

    if midnight_indices[0] != 0:
        intervals.append({
            "start_idx": 0, "end_idx": midnight_indices[0],
            "start_ts": all_ts[0], "end_ts": all_ts[midnight_indices[0]],
        })

    for k in range(len(midnight_indices) - 1):
        s, e = midnight_indices[k], midnight_indices[k + 1]
        intervals.append({
            "start_idx": s, "end_idx": e,
            "start_ts": all_ts[s], "end_ts": all_ts[e],
        })

    last_mid = midnight_indices[-1]
    if last_mid < ntimes - 1:
        intervals.append({
            "start_idx": last_mid, "end_idx": ntimes - 1,
            "start_ts": all_ts[last_mid], "end_ts": all_ts[-1],
        })

    return intervals


# =============================================================================
#  MAIN PIPELINE
# =============================================================================

def run_kerala_rainfall_pipeline(wrf_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    daily_dir = os.path.join(output_dir, "daily_rainfall_kerala")
    os.makedirs(daily_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. LOAD DATA
    # -------------------------------------------------------------------------
    print("[1/4] Loading WRF file and shapefiles...")
    ncfile = Dataset(wrf_path)
    counties, states, kerala_districts, kerala_state = load_geodata()
    cmap, norm = build_colormap()

    rain_t0          = getvar(ncfile, "RAINC", timeidx=0) + getvar(ncfile, "RAINNC", timeidx=0)
    lats, lons       = latlon_coords(rain_t0)
    ntimes       = ncfile.dimensions["Time"].size
    start_ts    = pd.to_datetime(str(getvar(ncfile, "Times", timeidx=0).values))
    end_ts      = pd.to_datetime(str(getvar(ncfile, "Times", timeidx=-1).values))
    delta_hours = int((end_ts - start_ts).total_seconds() // 3600)

    # -------------------------------------------------------------------------
    # 2. FULL CUMULATIVE — KERALA
    # -------------------------------------------------------------------------
    print("[2/4] Generating Kerala cumulative rainfall map...")

    rain_end   = getvar(ncfile, "RAINC", timeidx=-1) + getvar(ncfile, "RAINNC", timeidx=-1)
    rain_total = np.ma.masked_less(to_np(rain_end) - to_np(rain_t0), 1.0)

    fig, ax = plt.subplots(figsize=(8, 14), subplot_kw={"projection": ccrs.PlateCarree()})
    #  MAIN PIPELINE
# =============================================================================

def run_kerala_rainfall_pipeline(wrf_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    daily_dir = os.path.join(output_dir, "daily_rainfall_kerala")
    os.makedirs(daily_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. LOAD DATA
    # -------------------------------------------------------------------------
    print("[1/4] Loading WRF file and shapefiles...")
    ncfile = Dataset(wrf_path)
    counties, states, kerala_districts, kerala_state = load_geodata()
    cmap, norm = build_colormap()

    rain_t0          = getvar(ncfile, "RAINC", timeidx=0) + getvar(ncfile, "RAINNC", timeidx=0)
    lats, lons       = latlon_coords(rain_t0)
    ntimes       = ncfile.dimensions["Time"].size
    start_ts    = pd.to_datetime(str(getvar(ncfile, "Times", timeidx=0).values))
    end_ts      = pd.to_datetime(str(getvar(ncfile, "Times", timeidx=-1).values))
    delta_hours = int((end_ts - start_ts).total_seconds() // 3600)

    # -------------------------------------------------------------------------
    # 2. FULL CUMULATIVE — KERALA
    # -------------------------------------------------------------------------
    print("[2/4] Generating Kerala cumulative rainfall map...")

    rain_end   = getvar(ncfile, "RAINC", timeidx=-1) + getvar(ncfile, "RAINNC", timeidx=-1)
    rain_total = np.ma.masked_less(to_np(rain_end) - to_np(rain_t0), 1.0)

    fig, ax = plt.subplots(figsize=(8, 14), subplot_kw={"projection": ccrs.PlateCarree()})
    make_kerala_plot(
        fig, ax, lons, lats, rain_total, cmap, norm,
        param_label="RAINFALL (mm)",
        forecast_hours=delta_hours,
        based_on_ts=start_ts,
        valid_start_ts=start_ts,
        valid_end_ts=end_ts,
        kerala_districts=kerala_districts,
        kerala_state=kerala_state
    )
    add_weatherex_logo(fig)
    
    out_cum = os.path.join(output_dir, "kerala_cumulative_rainfall.png")
    plt.savefig(out_cum, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out_cum}")

    # -------------------------------------------------------------------------
    # 3. PER-DAY PLOTS — KERALA  (00 UTC → 00 UTC next day)
    # -------------------------------------------------------------------------
    day_intervals = get_day_intervals(ncfile, ntimes)
    print(f"[3/4] Generating {len(day_intervals)} daily Kerala plot(s)...")

    for day_num, interval in enumerate(day_intervals, start=1):
        s_idx   = interval["start_idx"]
        e_idx   = interval["end_idx"]
        s_ts    = interval["start_ts"]
        e_ts    = interval["end_ts"]
        day_hrs = int((e_ts - s_ts).total_seconds() // 3600)

        r_start = to_np(getvar(ncfile, "RAINC",  timeidx=s_idx) +
                        getvar(ncfile, "RAINNC", timeidx=s_idx))
        r_end   = to_np(getvar(ncfile, "RAINC",  timeidx=e_idx) +
                        getvar(ncfile, "RAINNC", timeidx=e_idx))
        daily   = np.ma.masked_less(r_end - r_start, 1.0)

        fig, ax = plt.subplots(
            figsize=(8, 14),
            subplot_kw={"projection": ccrs.PlateCarree()}
        )
        make_kerala_plot(
            fig, ax, lons, lats, daily, cmap, norm,
            param_label="RAINFALL (mm)",
            forecast_hours=day_hrs,
            based_on_ts=start_ts,
            valid_start_ts=s_ts,
            valid_end_ts=e_ts,
            kerala_districts=kerala_districts,
            kerala_state=kerala_state
        )
        add_weatherex_logo(fig)
    
      

        fname    = f"kerala_daily_Day{day_num}_{s_ts.strftime('%d%b%Y')}.png"
        out_path = os.path.join(daily_dir, fname)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"    Day {day_num}: {s_ts.strftime('%d %b %Y %H UTC')} → "
              f"{e_ts.strftime('%d %b %Y %H UTC')}  →  {fname}")

    # -------------------------------------------------------------------------
    # 4. HOURLY ANIMATION — KERALA
    # -------------------------------------------------------------------------
    print("[4/4] Generating Kerala hourly animation...")

    fig2, ax2 = plt.subplots(
        figsize=(8, 14),
        subplot_kw={"projection": ccrs.PlateCarree()}
    )
    apply_kerala_map_features(ax2, kerala_districts, kerala_state)
    add_weatherex_logo(fig2)

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

        '''cf_new = ax2.contourf(
            to_np(lons), to_np(lats), hourly,
            levels=LEVELS, cmap=cmap, norm=norm,
            extend="max", transform=ccrs.PlateCarree(),
            zorder=2, alpha=0.85
        )'''
        cf_new = ax2.pcolormesh(
        to_np(lons),
        to_np(lats),
        hourly,
        cmap=cmap,
        norm=norm,
        shading="auto",
        transform=ccrs.PlateCarree(),
        zorder=2,
        alpha=0.85
        )
        _current_cf[0] = cf_new

        t_prev = pd.to_datetime(str(getvar(ncfile, "Times", timeidx=frame).values))
        t_now  = pd.to_datetime(str(getvar(ncfile, "Times", timeidx=frame+1).values))
        set_title(ax2,
            param_label="RAINFALL (mm)",
            forecast_hours=1,
            based_on_ts=start_ts,
            valid_start_ts=t_prev,
            valid_end_ts=t_now,
        )
        return []

    ani = animation.FuncAnimation(
        fig2, update, frames=ntimes - 1,
        interval=1500, blit=False
    )
    out_ani = os.path.join(output_dir, "kerala_hourly_animation.mp4")
    ani.save(out_ani, writer=FFMpegWriter(fps=1), dpi=150)
    plt.close()
    print(f"    Saved: {out_ani}")

    ncfile.close()
    print("\nDone! All Kerala outputs saved to:", output_dir)
    print(f"  • kerala_cumulative_rainfall.png")
    print(f"  • kerala_hourly_animation.mp4")
    print(f"  • daily_rainfall_kerala/  ({len(day_intervals)} daily PNG files)")
    add_weatherex_logo(fig)
    
    out_cum = os.path.join(output_dir, "kerala_cumulative_rainfall.png")
    plt.savefig(out_cum, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out_cum}")

    # -------------------------------------------------------------------------
    # 3. PER-DAY PLOTS — KERALA  (00 UTC → 00 UTC next day)
    # -------------------------------------------------------------------------
    day_intervals = get_day_intervals(ncfile, ntimes)
    print(f"[3/4] Generating {len(day_intervals)} daily Kerala plot(s)...")

    for day_num, interval in enumerate(day_intervals, start=1):
        s_idx   = interval["start_idx"]
        e_idx   = interval["end_idx"]
        s_ts    = interval["start_ts"]
        e_ts    = interval["end_ts"]
        day_hrs = int((e_ts - s_ts).total_seconds() // 3600)

        r_start = to_np(getvar(ncfile, "RAINC",  timeidx=s_idx) +
                        getvar(ncfile, "RAINNC", timeidx=s_idx))
        r_end   = to_np(getvar(ncfile, "RAINC",  timeidx=e_idx) +
                        getvar(ncfile, "RAINNC", timeidx=e_idx))
        daily   = np.ma.masked_less(r_end - r_start, 1.0)

        fig, ax = plt.subplots(
            figsize=(8, 14),
            subplot_kw={"projection": ccrs.PlateCarree()}
        )
        make_kerala_plot(
            fig, ax, lons, lats, daily, cmap, norm,
            param_label="RAINFALL (mm)",
            forecast_hours=day_hrs,
            based_on_ts=s_ts,
            #based_on_str=fmt_bfs_time_ts(s_ts),
            #valid_for_str=fmt_bfs_time_ts(e_ts),
            kerala_districts=kerala_districts,
            kerala_state=kerala_state
          
        )
        add_weatherex_logo(fig)
    
      

        fname    = f"kerala_daily_Day{day_num}_{s_ts.strftime('%d%b%Y')}.png"
        out_path = os.path.join(daily_dir, fname)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"    Day {day_num}: {s_ts.strftime('%d %b %Y %H UTC')} → "
              f"{e_ts.strftime('%d %b %Y %H UTC')}  →  {fname}")

    # -------------------------------------------------------------------------
    # 4. HOURLY ANIMATION — KERALA
    # -------------------------------------------------------------------------
    print("[4/4] Generating Kerala hourly animation...")

    fig2, ax2 = plt.subplots(
        figsize=(8, 14),
        subplot_kw={"projection": ccrs.PlateCarree()}
    )
    apply_kerala_map_features(ax2, kerala_districts, kerala_state)
    add_weatherex_logo(fig2)

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

        '''cf_new = ax2.contourf(
            to_np(lons), to_np(lats), hourly,
            levels=LEVELS, cmap=cmap, norm=norm,
            extend="max", transform=ccrs.PlateCarree(),
            zorder=2, alpha=0.85
        )'''
        cf_new = ax2.pcolormesh(
        to_np(lons),
        to_np(lats),
        hourly,
        cmap=cmap,
        norm=norm,
        shading="auto",
        transform=ccrs.PlateCarree(),
        zorder=2,
        alpha=0.85
        )
        _current_cf[0] = cf_new

        t_prev = pd.to_datetime(str(getvar(ncfile, "Times", timeidx=frame).values))
        t_now  = pd.to_datetime(str(getvar(ncfile, "Times", timeidx=frame+1).values))
        set_title(ax2,
            param_label="RAINFALL (mm)",
            forecast_hours=1,
            based_on_ts=t_prev
            #based_on_str=fmt_bfs_time_ts(t_prev),
            #valid_for_str=fmt_bfs_time_ts(t_now)
        )
        return []

    ani = animation.FuncAnimation(
        fig2, update, frames=ntimes - 1,
        interval=1500, blit=False
    )
    out_ani = os.path.join(output_dir, "kerala_hourly_animation.mp4")
    ani.save(out_ani, writer=FFMpegWriter(fps=1), dpi=150)
    plt.close()
    print(f"    Saved: {out_ani}")

    ncfile.close()
    print("\nDone! All Kerala outputs saved to:", output_dir)
    print(f"  • kerala_cumulative_rainfall.png")
    print(f"  • kerala_hourly_animation.mp4")
    print(f"  • daily_rainfall_kerala/  ({len(day_intervals)} daily PNG files)")


# =============================================================================
#  ENTRY POINT
# =============================================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="NWP Kerala Rainfall output"
    )

    today = datetime.now().strftime("%Y-%m-%d")

    DEFAULT_INPUT = (
        f"/home/ras_08/Models/WRF_TUTORIAL/WRFV4.5/run/"
        f"wrfout_d01_{today}_00:00:00"
    )

    DEFAULT_OUTDIR = (
        f"/home/ras_08/WEATHER/INDIA/Data/"
        f"{datetime.now().strftime('%Y%m%d')}/00z"
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

    run_kerala_rainfall_pipeline(
        args.input,
        args.outdir
    )
