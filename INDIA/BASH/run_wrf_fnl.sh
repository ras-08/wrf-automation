
# --- 1. CONFIGURATION & PATHS ---
#PASSWORD="YOUR_PASSWORD"
CONDA_ENV="ncl_stable"

HOME_DIR="/home/ras_08"

WEATHER_DIR="$HOME_DIR/WEATHER"

WPS_DIR="$HOME_DIR/Models/WRF_TUTORIAL/WPS-4.5"

WRF_RUN_DIR="$HOME_DIR/Models/WRF_TUTORIAL/WRFV4.5/run"

WORKFLOW_DIR="$WEATHER_DIR/INDIA/WORKFLOW"

DATA_ROOT="$WEATHER_DIR/INDIA/Data"

FNL_ROOT="$WEATHER_DIR/INDIA/FNL-DATA"


# Dynamic Dates
TODAY_DASH=$(date +%Y-%m-%d)
TODAY_NODASH=$(date +%Y%m%d)
TOMORROW_DASH=$(date -d "tomorrow" +%Y-%m-%d)
END_DASH=$(date -d "+3 days" +%Y-%m-%d)
YES_DASH=$(date -d "yesterday" +%Y-%m-%d)
YES_NODASH=$(date -d "yesterday" +%Y%m%d)


FNL_FULL_PATH="$FNL_ROOT/FNL_INDIA_${TODAY_NODASH}"
#GFS_FULL_PATH="$GFS_ROOT/GFS_INDIA_${TODAY_NODASH}_00z"
DAILY_PLOT_DIR="$DATA_ROOT/$TODAY_NODASH"

echo "------------------------------------------------"
echo "STARTING COMPLETE WEATHER PIPELINE FOR $TODAY_DASH"
echo "------------------------------------------------"

# --- 2. CONDA INITIALIZATION ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ncl_stable

# --- 3. SYSTEM UPDATES ---
#echo "[1/7] Updating System..."
#echo "$PASSWORD" | sudo -S apt update && echo "$PASSWORD" | sudo -S apt upgrade -y && echo "$PASSWORD" | sudo -S apt autoremove -y

#--- 4. FNL  DATA DOWNLOAD ---
echo "[2/7] Downloading FNL to $FNL_FULL_PATH..."
mkdir -p "$FNL_FULL_PATH"
python "$WORKFLOW_DIR/ncep_downloader.py" --date "$TODAY_NODASH" --outdir "$FNL_FULL_PATH"

#changing to simple grib2
SIMPLE_DIR="$FNL_FULL_PATH/simple"
mkdir -p "$SIMPLE_DIR"

for f in "$FNL_FULL_PATH"/*oper-fc.grib2; do
    base=$(basename "$f")
    echo "Converting $base ..."
    grib_set -r -s packingType=grid_simple "$f" "$SIMPLE_DIR/$base"
done
# --- 5. WPS PROCESSING ---
echo "[3/7] Running WPS (Ungrib & Metgrid)..."
cd "$WPS_DIR" || exit
rm -f FILE:* GRIBFILE.* met_em.d01.* geo_em.d01.nc metgrid.log ungrib.log  geogrid.log

csh ./link_grib.csh "$SIMPLE_DIR/"*oper-fc.grib2

# Update Namelist.wps
sed -i "s/start_date = .*/start_date = '${TODAY_DASH}_00:00:00',/g" namelist.wps
sed -i "s/end_date = .*/end_date = '${TOMORROW_DASH}_00:00:00',/g" namelist.wps
#sed -i "s/end_date = .*/end_date = '${END_DASH}_00:00:00',/g" namelist.wps
#sed -i "s/end_date = .i*/end_date = '${TODAY_DASH}_00:00:00',/g" namelist.wps
./geogrid.exe > log.geogrid
ln -sf ungrib/Variable_Tables/Vtable.ECMWF.1 Vtable
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
sed -i "s/start_year.*/start_year = $(date +%Y),/g" namelist.input
sed -i "s/start_month.*/start_month = $(date +%m),/g" namelist.input
sed -i "s/start_day.*/start_day = $(date +%d),/g" namelist.input
sed -i "s/end_year.*/end_year = $(date -d tomorrow +%Y),/g" namelist.input
sed -i "s/end_month.*/end_month = $(date -d tomorrow +%m),/g" namelist.input
sed -i "s/end_day.*/end_day = $(date -d tomorrow +%d),/g" namelist.input

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
fi

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
    
    echo "[9/9] Plotting kerala_rainfall for specified locations in the script"
    python "$WORKFLOW_DIR/plot_kerala_rainfall.py" --input "$WRFOUT_FILE"

    echo "------------------------------------------------"
    echo "PIPELINE COMPLETE! Graphics are in $DAILY_PLOT_DIR"
    echo "------------------------------------------------"
else
    echo "ERROR: WRF output file $WRFOUT_FILE not found. Plotting skipped."
    exit 1
fi

