#!/bin/bash

# --- 1. CONFIGURATION & PATHS ---
#PASSWORD="YOUR_PASSWORD"
CONDA_ENV="ncl_stable"

HOME_DIR="/home/ras_08"

WEATHER_DIR="$HOME_DIR/WEATHER"

WPS_DIR="$HOME_DIR/Models/WRF_TUTORIAL/WPS-4.5"

WRF_RUN_DIR="$HOME_DIR/Models/WRF_TUTORIAL/WRFV4.5/run"

WORKFLOW_DIR="$WEATHER_DIR/INDIA/WORKFLOW"

DATA_ROOT="$WEATHER_DIR/INDIA/Data"

GFS_ROOT="$WEATHER_DIR/INDIA/GFS-DATA"

# Dynamic Dates
TODAY_DASH=$(date +%Y-%m-%d)
END_DASH=$(date -d "+tomorrow" +%Y-%m-%d)
#END_DASH=$(date -d "+2 days" +%Y-%m-%d)

TODAY_NODASH=$(date +%Y%m%d)
#YES_DASH=$(date -%Y-%m-%d)
#YES_NODASH=$(date +%Y-%m-%d)

GFS_FULL_PATH="$GFS_ROOT/GFS_INDIA_${TODAY_NODASH}_00z"
#GFS_FULL_PATH="$GFS_ROOT/GFS_INDIA_${YES_NODASH}_00z"
DAILY_PLOT_DIR="$DATA_ROOT/$TODAY_NODASH"
START_YEAR=$(date -d "$TODAY_DASH" +%Y)
START_MONTH=$(date -d "$TODAY_DASH" +%m)
START_DAY=$(date -d "$TODAY_DASH" +%d)
END_YEAR=$(date -d "$END_DASH" +%Y)
END_MONTH=$(date -d "$END_DASH" +%m)
END_DAY=$(date -d "$END_DASH" +%d)

echo "Today: $TODAY_DASH"
echo "End:   $END_DASH"

echo "------------------------------------------------"
echo "STARTING COMPLETE WEATHER PIPELINE FOR $TODAY_DASH"
echo "------------------------------------------------"

# --- 2. CONDA INITIALIZATION ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ncl_stable


# --- 3. SYSTEM UPDATES ---
#echo "[1/7] Updating System..."
#echo "$PASSWORD" | sudo -S apt update && echo "$PASSWORD" | sudo -S apt upgrade -y && echo "$PASSWORD" | sudo -S apt autoremove -y
'''
# --- 4. GFS DATA DOWNLOAD ---echo "[2/7] Downloading GFS to $GFS_FULL_PATH..."
mkdir -p "$GFS_FULL_PATH"
python "$WORKFLOW_DIR/gfs_downloader.py" --date "$TODAY_NODASH" --outdir "$GFS_FULL_PATH"

# --- 5. WPS PROCESSING ---
echo "[3/7] Running WPS (Ungrib & Metgrid)..."
cd "$WPS_DIR" || exit
rm -f FILE:* GRIBFILE.* met_em.d01.* metgrid.log ungrib.log 

csh ./link_grib.csh "$GFS_FULL_PATH/gfs.t00z.pgrb2.0p25"*

sed -i "s/^[[:space:]]*start_date.*/start_date = '${TODAY_DASH}_00:00:00',/" namelist.wps
sed -i "s/^[[:space:]]*end_date.*/end_date   = '${END_DASH}_00:00:00',/" namelist.wps


./geogrid.exe > log.geogrid
ln -sf ungrib/Variable_Tables/Vtable.GFS Vtable
./ungrib.exe > log.ungrib
./metgrid.exe > log.metgrid

# --- 6. WRF PROCESSING ---
echo "[4/7] Running WRF Processing..."
cd "$WRF_RUN_DIR" || exit

# Clean previous run files
rm -f met_em.d01.* wrfout_d01_* wrfbdy_d01 wrfinput_d01 rsl.*

# Link current met_em files
ln -sf "$WPS_DIR"/met_em.d01.* .

# Update Namelist.input dates
sed -i "s/start_year.*/start_year = ${START_YEAR},/g" namelist.input
sed -i "s/start_month.*/start_month = ${START_MONTH},/g" namelist.input
sed -i "s/start_day.*/start_day = ${START_DAY},/g" namelist.input
#CHANGE THIS BLOCK  ACCORDINGLY
sed -i "s/start_hour.*/start_hour = 00,/g" namelist.input

sed -i "s/end_year.*/end_year = ${END_YEAR},/g" namelist.input
sed -i "s/end_month.*/end_month = ${END_MONTH},/g" namelist.input
sed -i "s/end_day.*/end_day = ${END_DAY},/g" namelist.input
#CHANGE THIS BLOCK ACCORDINGLY
sed -i "s/end_hour.*/end_hour = 00,/g" namelist.input

echo "Executing real.exe..."
mpirun -np 8 ./real.exe

# Check real.exe success before proceeding
if grep -q "SUCCESS COMPLETE REAL_EM" rsl.error.0000; then
    echo "real.exe successful. Starting wrf.exe..."
    
    # Run WRF in the background
    mpirun -np 8 ./wrf.exe & 
    WRF_PID=$!
    
    echo "--- Monitoring WRF Progress (rsl.error.0000) ---"
    tail -f rsl.error.0000 --pid=$WRF_PID
    
    echo "--- WRF Run Finished ---"
else
    echo "ERROR: real.exe failed. Check $WRF_RUN_DIR/rsl.error.0000"
    exit 1
fi'''

# --- 7. PLOTTING ---
mkdir -p "$DAILY_PLOT_DIR"
WRFOUT_FILE="$WRF_RUN_DIR/wrfout_d01_${TODAY_DASH}_00:00:00"


# --- 7. PLOTTING ---
mkdir -p "$DAILY_PLOT_DIR"
#WRFOUT_FILE="$WRF_RUN_DIR/wrfout_d01_${TODAY_DASH}_00:00:00"
WRFOUT_FILE="$WRF_RUN_DIR/wrfout_d01_${TODAY_DASH}_00:00:00"


# Check if wrfout file exists before plotting
if [ -f "$WRFOUT_FILE" ]; then
    echo "[5/9] Plotting Total Rainfall..."
    python "$WORKFLOW_DIR/plot_rainfall_bfs.py" --input "$WRFOUT_FILE" --outdir "$DAILY_PLOT_DIR"

    echo "[6/9] Plotting Maximum Temperature..."
    python "$WORKFLOW_DIR/plot_max_temp.py" --input "$WRFOUT_FILE" --outdir "$DAILY_PLOT_DIR"

    echo "[7/9] Plotting Minimum Temperature..."
    python "$WORKFLOW_DIR/plot_min_temp.py" --input "$WRFOUT_FILE" --outdir "$DAILY_PLOT_DIR"
    
    echo "[8/9] PLotting Maximum Wind ..."
    python "$WORKFLOW_DIR/plot_wind.py"  --input "$WRFOUT_FILE"  --outdir "$DAILY_PLOT_DIR" 
    
    echo "[9/9] Plotting SkweT Log-P plotos for specified locations in the script"
    python "$WORKFLOW_DIR/plot_skewT.py" --input "$WRFOUT_FILE"
    
    echo "[10/10] Plotting kerala_rainfall for specified locations in the script"
    python "$WORKFLOW_DIR/plot_kerala_rainfall.py" --input "$WRFOUT_FILE"
    
    echo "[10/10] Plotting uttarakhand_rainfall for specified locations in the script"
    python "$WORKFLOW_DIR/plot_uttarakhand_rainfall.py" --input "$WRFOUT_FILE"
    
    echo "[10/10] Plotting maharashtra_rainfall for specified locations in the script"
    python "$WORKFLOW_DIR/plot_maharashtra_rainfall.py" --input "$WRFOUT_FILE"
    
    
    echo "------------------------------------------------"
    echo "PIPELINE COMPLETE! Graphics are in $DAILY_PLOT_DIR"
    echo "------------------------------------------------"
else
    echo "ERROR: WRF output file $WRFOUT_FILE not found. Plotting skipped."
    exit 1
fi
