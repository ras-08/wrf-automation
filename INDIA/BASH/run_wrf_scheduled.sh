#!/bin/bash
# run_wrf_scheduled.sh
# Determines cycle (00 or 12) from current UTC time, polls until data
# available, runs pipeline, exits.

NOW_H=$(date -u +%H)
NOW_M=$(date -u +%M)
NOW_MIN=$(( 10#$NOW_H * 60 + 10#$NOW_M ))

if (( NOW_MIN >= 240 && NOW_MIN <= 510 )); then
    CYCLE=00; END_MIN=510       # 04:00-08:30 UTC (IST 09:30-14:00)
elif (( NOW_MIN >= 960 && NOW_MIN <= 1109 )); then
    CYCLE=12; END_MIN=1109      # 16:00-18:29 UTC (IST 21:30-23:59)
else
    echo "Outside any cycle window. Exiting."
    exit 0
fi

echo "Detected cycle: ${CYCLE}z  (poll until minute $END_MIN UTC)"

POLL_INTERVAL=600   # 10 minutes

while true; do
    CUR_MIN=$(( 10#$(date -u +%H) * 60 + 10#$(date -u +%M) ))
    if (( CUR_MIN > END_MIN )); then
        echo "Window closed without successful run. Exiting."
        exit 1
    fi
    /home/ras_08/WEATHER/INDIA/BASH/run_wrf_auto.sh "$CYCLE"
    if [[ $? -eq 0 ]]; then
        echo "Pipeline succeeded."
        exit 0
    fi
    echo "Not ready / failed. Retrying in $((POLL_INTERVAL/60)) min ..."
    sleep $POLL_INTERVAL
done
