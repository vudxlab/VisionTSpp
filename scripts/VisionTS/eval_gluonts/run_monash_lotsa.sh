for ds in m1_monthly monash_m3_monthly monash_m3_other m4_monthly m4_weekly m4_daily m4_hourly tourism_quarterly tourism_monthly cif_2016_6 cif_2016_12 australian_electricity_demand bitcoin_with_missing pedestrian_counts vehicle_trips_with_missing kdd_cup_2018_with_missing weather nn5_daily_with_missing nn5_weekly car_parts_with_missing fred_md traffic_hourly traffic_weekly rideshare_with_missing hospital covid_deaths temperature_rain_with_missing sunspot_with_missing saugeenday us_births
do
  python run_std.py --dataset=${ds} --save_name=visionts.csv --periodicity=autotune --context_len=1000 --no_periodicity_context_len=300 --batch_size 128
  # python run.py --dataset=${ds} --save_name=${1}_monash.csv --periodicity=autotune --context_len=1000 --no_periodicity_context_len=300 --batch_size 128
done
