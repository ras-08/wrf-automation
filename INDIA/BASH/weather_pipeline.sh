#!/bin/bash

# =============================================================================
# COMPLETE WRF WEATHER PIPELINE
# GFS folder : yesterday's download (GFS_INDIA_20260605_00z)
# Forecast   : today (6 June 2026) --> +3 days (9 June 2026)
# =============================================================================

# --- 1. CONFIGURATION & PATHS ---
CONDA_ENV="ncl_stable"
HOME_DIR="/home/ras_08"
WEATHER_DIR="$HOME_DIR/WEATHER"
WPS_DIR="$HOME_DIR/Models/WRF_TUTORIAL/WPS-4.5"
WRF_RUN_DIR="$HOME_DIR/Models/WRF_TUTORIAL/WRFV4.5/run"
WORKFLOW_DIR="$WEATHER_DIR/INDIA/WORKFLOW"
DATA_ROOT="$WEATHER_DIR/INDIA/Data"
GFS_ROOT="$WEATHER_DIR/INDIA/GFS-DATA"

# --- 2. DYNAMIC DATES ---
TODAY_DASH=$(date +%Y-%m-%d)                     # 2026-06-06  → WRF start / namelist dates
TODAY_NODASH=$(date +%Y%m%d)                     # 20260606    → plot output folder

YES_NODASH=$(date -d "yesterday" +%Y%m%d)        # 20260605    → GFS folder name only

END_DASH=$(date -d "+3 days" +%Y-%m-%d)          # 2026-06-09  → WRF end

# GFS folder = yesterday's downloaded data
GFS_FULL_PATH="$GFS_ROOT/GFS_INDIA_${YES_NODASH}_00z"

# Plot output folder = today's date
DAILY_PLOT_DIR="$DATA_ROOT/$TODAY_NODASH"

echo "================================================"
echo " STARTING COMPLETE WEATHER PIPELINE"
echo "  GFS source (yesterday's folder) : $GFS_FULL_PATH"
echo "  WRF start  (today)              : $TODAY_DASH 00:00:00"
echo "  WRF end    (+3 days)            : $END_DASH 00:00:00"
echo "  Plot output dir                 : $DAILY_PLOT_DIR"
echo "================================================"

# --- 3. CONDA INITIALIZATION ---
echo "[INIT] Activating conda environment: $CONDA_ENV"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

# --- 4. SYSTEM UPDATES (optional — uncomment to enable) ---
# echo "[1/7] Updating system packages..."
# echo "$PASSWORD" | sudo -S apt update
# echo "$PASSWORD" | sudo -S apt upgrade -y
# echo "$PASSWORD" | sudo -S apt autoremove -y

# --- 5. GFS DATA DOWNLOAD (SKIPPED — using yesterday's pre-downloaded data) ---
# To re-enable, uncomment below:
#
# echo "[2/7] Downloading GFS data to $GFS_FULL_PATH..."
# mkdir -p "$GFS_FULL_PATH"
# python "$WORKFLOW_DIR/gfs_downloader.py" --date "$YES_NODASH" --outdir "$GFS_FULL_PATH"

echo "[2/7] GFS download SKIPPED — using pre-downloaded data at: $GFS_FULL_PATH"

# Validate GFS directory exists and has files
if [ ! -d "$GFS_FULL_PATH" ]; then
    echo "ERROR: GFS directory not found: $GFS_FULL_PATH"
    exit 1
fi
if [ -z "$(ls -A "$GFS_FULL_PATH" 2>/dev/null)" ]; then
    echo "ERROR: GFS directory is empty: $GFS_FULL_PATH"
    exit 1
fi
echo "  GFS data validated OK: $GFS_FULL_PATH"

# --- 6. WPS PROCESSING ---
echo "[3/7] Running WPS (Geogrid, Ungrib, Metgrid)..."
cd "$WPS_DIR" || { echo "ERROR: Cannot cd to $WPS_DIR"; exit 1; }

# Clean previous run artifacts
rm -f FILE:* GRIBFILE.* met_em.d01.* geo_em.d01.nc \
      metgrid.log ungrib.log geogrid.log \
      log.geogrid log.ungrib log.metgrid

# Link yesterday's GRIB files (folder = yesterday, data covers today onwards)
csh ./link_grib.csh "$GFS_FULL_PATH/gfs.t00z.pgrb2.0p25"*

# Update namelist.wps
#   start_date = today 00z      (forecast start)
#   end_date   = today +3 days  (forecast end)
sed -i "s/start_date = .*/start_date = '${TODAY_DASH}_00:00:00',/g" namelist.wps
sed -i "s/end_date = .*/end_date = '${END_DASH}_00:00:00',/g"       namelist.wps

echo "  namelist.wps → start: ${TODAY_DASH}_00:00:00 | end: ${END_DASH}_00:00:00"

echo "  Running geogrid.exe..."
./geogrid.exe > log.geogrid 2>&1

echo "  Running ungrib.exe..."
./ungrib.exe > log.ungrib 2>&1

echo "  Running metgrid.exe..."
./metgrid.exe > log.metgrid 2>&1

# Validate metgrid produced output
if ! ls met_em.d01.* 1>/dev/null 2>&1; then
    echo "ERROR: WPS produced no met_em files. Check $WPS_DIR/log.metgrid"
    exit 1
fi
echo "  WPS completed OK."

# --- 7. WRF PROCESSING ---
echo "[4/7] Running WRF (real.exe + wrf.exe)..."
cd "$WRF_RUN_DIR" || { echo "ERROR: Cannot cd to $WRF_RUN_DIR"; exit 1; }

# Clean previous run files
rm -f met_em.d01.* wrfout_d01_* wrfbdy_d01 wrfinput_d01 rsl.*

# Link met_em files from WPS
ln -sf "$WPS_DIR"/met_em.d01.* .

# Update namelist.input
#   start = today     (forecast start)
#   end   = today +3  (forecast end)
sed -i "s/start_year.*/start_year  = $(date +%Y),/g"                namelist.input
sed -i "s/start_month.*/start_month = $(date +%m),/g"               namelist.input
sed -i "s/start_day.*/start_day   = $(date +%d),/g"                 namelist.input
sed -i "s/end_year.*/end_year    = $(date -d "+3 days" +%Y),/g"     namelist.input
sed -i "s/end_month.*/end_month   = $(date -d "+3 days" +%m),/g"    namelist.input
sed -i "s/end_day.*/end_day     = $(date -d "+3 days" +%d),/g"      namelist.input

echo "  namelist.input → start: $TODAY_DASH | end: $END_DASH"

echo "  Executing real.exe..."
mpirun -np 8 ./real.exe

# Check real.exe success
if grep -q "SUCCESS COMPLETE REAL_EM" rsl.error.0000; then
    echo "  real.exe successful. Launching wrf.exe..."

    mpirun -np 8 ./wrf.exe &
    WRF_PID=$!

    echo "--- Monitoring WRF progress (rsl.error.0000) ---"
    tail -f rsl.error.0000 --pid=$WRF_PID

    wait $WRF_PID
    echo "--- WRF run finished ---"
else
    echo "ERROR: real.exe failed. Check $WRF_RUN_DIR/rsl.error.0000"
    exit 1
fi

# --- 8. PLOTTING ---
echo "[5/9] Setting up plot output directory..."
mkdir -p "$DAILY_PLOT_DIR"

# wrfout file is stamped with WRF start date (today)
WRFOUT_FILE="$WRF_RUN_DIR/wrfout_d01_${TODAY_DASH}_00:00:00"

if [ ! -f "$WRFOUT_FILE" ]; then
    echo "ERROR: WRF output file not found: $WRFOUT_FILE"
    echo "  Plotting skipped. Check WRF run logs."
    exit 1
fi
echo "  wrfout file found: $WRFOUT_FILE"

echo "[5/9] Plotting Total Rainfall..."
python "$WORKFLOW_DIR/plot_rainfall.py" --input "$WRFOUT_FILE" --outdir "$DAILY_PLOT_DIR"

echo "[6/9] Plotting Maximum Temperature..."
python "$WORKFLOW_DIR/plot_max_temp.py" --input "$WRFOUT_FILE" --outdir "$DAILY_PLOT_DIR"

echo "[7/9] Plotting Minimum Temperature..."
python "$WORKFLOW_DIR/plot_min_temp.py" --input "$WRFOUT_FILE" --outdir "$DAILY_PLOT_DIR"

echo "[8/9] Plotting Maximum Wind..."
python "$WORKFLOW_DIR/plot_wind.py" --input "$WRFOUT_FILE" --outdir "$DAILY_PLOT_DIR"

echo "[9/9] Plotting Skew-T Log-P for specified locations..."
python "$WORKFLOW_DIR/plot_skewT.py" --input "$WRFOUT_FILE"

echo "================================================"
echo " PIPELINE COMPLETE!"
echo "  Graphics saved to: $DAILY_PLOT_DIR"
echo "================================================"
