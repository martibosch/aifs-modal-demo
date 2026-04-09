"""Snakemake pipeline — AIFS storm track predictability for Storm Claudia."""

from datetime import datetime, timedelta
from pathlib import Path

PAPERMILL = "papermill -k python3"

NOTEBOOKS_DIR = Path("notebooks")
NOTEBOOKS_OUTPUT_DIR = NOTEBOOKS_DIR / "output"
DATA_DIR = Path("data")
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_INTERIM_DIR = DATA_DIR / "interim"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

########################################################################################
# Storm Claudia
STORM_NAME = "claudia"
NAMING_DATE = "2025-11-10"
WINDOW_DAYS = 7  # ±7 days around naming date for ERA5 download

# ERA5 peak intensity: 2025-11-08 00 UTC (track_id=51, min MSL anomaly)
ERA5_PEAK = datetime(2025, 11, 8, 0)
ERA5_TRACK_ID = 51

########################################################################################
# AIFS forecast settings
AIFS_LEAD_TIME_HOURS = 168  # 7 days per forecast (shorter for demo)
# ACHTUNG: n_members=0 means deterministic forecast with the aifs-single-1.1 checkpoint
# whereas n_members=1 means an ensemble forecast with the aifs-ensemble-1.1 checkpoint
# and writes into a ensemble-shaped output schema
AIFS_N_MEMBERS = 0

AIFS_STORAGE_BUCKET = "aifs-modal-unibe"
AIFS_ICS_PREFIX = "aifs-ic-ifs"
AIFS_ICS_BRANCH = "main"
AIFS_OUTPUTS_PREFIX = "aifs-outputs-det"

# D-1 through D-7 from ERA5 peak
INIT_DATES = [(ERA5_PEAK - timedelta(days=d)).strftime("%Y%m%d%H") for d in range(1, 8)]

########################################################################################
# Atlantic bounding box for dominant track selection
ATLANTIC_LAT = [25.0, 65.0]
ATLANTIC_LON = [-60.0, 10.0]


rule download_era5:
    input:
        notebook=NOTEBOOKS_DIR / "01-download-era5.ipynb",
    output:
        nc=DATA_RAW_DIR / f"msl_{STORM_NAME}_{WINDOW_DAYS}d_global_2p5deg.nc",
        notebook=NOTEBOOKS_OUTPUT_DIR / f"01-download-era5_{STORM_NAME}.ipynb",
    params:
        naming_date=NAMING_DATE,
        window_days=WINDOW_DAYS,
    shell:
        "{PAPERMILL} {input.notebook} {output.notebook} "
        "-p storm_name {STORM_NAME} "
        "-p naming_date_str {params.naming_date} "
        "-p window_days {params.window_days} "
        "-p output_file {output.nc}"


rule era5_tracking:
    input:
        nc=DATA_RAW_DIR / f"msl_{STORM_NAME}_{WINDOW_DAYS}d_global_2p5deg.nc",
        notebook=NOTEBOOKS_DIR / "02-era5-tracking.ipynb",
    output:
        csv=DATA_INTERIM_DIR / f"era5_tracks_{STORM_NAME}.csv",
        notebook=NOTEBOOKS_OUTPUT_DIR / f"02-era5-tracking_{STORM_NAME}.ipynb",
    params:
        era5_track_id=ERA5_TRACK_ID,
        atlantic_lat=ATLANTIC_LAT,
        atlantic_lon=ATLANTIC_LON,
    shell:
        "{PAPERMILL} {input.notebook} {output.notebook} "
        "-p storm_name {STORM_NAME} "
        "-p input_file {input.nc} "
        "-p era5_track_id {params.era5_track_id} "
        "-p output_file {output.csv}"


rule ingest_ic:
    input:
        notebook=NOTEBOOKS_DIR / "03-ingest-ic.ipynb",
    output:
        marker=DATA_INTERIM_DIR / f"aifs_ic_{STORM_NAME}_{{init_date}}.done",
        notebook=NOTEBOOKS_OUTPUT_DIR / f"03-ingest-ic_{STORM_NAME}_{{init_date}}.ipynb",
    params:
        storage_bucket=AIFS_STORAGE_BUCKET,
        initial_conditions_prefix=AIFS_ICS_PREFIX,
        initial_conditions_branch=AIFS_ICS_BRANCH,
    shell:
        "{PAPERMILL} {input.notebook} {output.notebook} "
        "-p storm_name {STORM_NAME} "
        "-p init_date_str {wildcards.init_date} "
        "-p storage_bucket {params.storage_bucket} "
        "-p initial_conditions_prefix {params.initial_conditions_prefix} "
        "-p initial_conditions_branch {params.initial_conditions_branch} "
        "-p marker_file {output.marker}"


rule all_ingest_ic:
    input:
        expand(
            DATA_INTERIM_DIR / f"aifs_ic_{STORM_NAME}_{{init_date}}.done",
            init_date=INIT_DATES,
        ),


rule run_aifs_forecast:
    input:
        ic_marker=DATA_INTERIM_DIR / f"aifs_ic_{STORM_NAME}_{{init_date}}.done",
        notebook=NOTEBOOKS_DIR / "03-run-aifs-forecast.ipynb",
    output:
        marker=DATA_INTERIM_DIR / f"aifs_forecast_{STORM_NAME}_{{init_date}}.done",
        notebook=NOTEBOOKS_OUTPUT_DIR
        / f"03-run-aifs-forecast_{STORM_NAME}_{{init_date}}.ipynb",
    resources:
        kernels=1,
    params:
        lead_time=AIFS_LEAD_TIME_HOURS,
        n_members=AIFS_N_MEMBERS,
        storage_bucket=AIFS_STORAGE_BUCKET,
        initial_conditions_prefix=AIFS_ICS_PREFIX,
        initial_conditions_branch=AIFS_ICS_BRANCH,
        outputs_prefix=AIFS_OUTPUTS_PREFIX,
    shell:
        "{PAPERMILL} {input.notebook} {output.notebook} "
        "-p storm_name {STORM_NAME} "
        "-p init_date_str {wildcards.init_date} "
        "-p lead_time {params.lead_time} "
        "-p n_members {params.n_members} "
        "-p storage_bucket {params.storage_bucket} "
        "-p initial_conditions_prefix {params.initial_conditions_prefix} "
        "-p initial_conditions_branch {params.initial_conditions_branch} "
        "-p outputs_prefix {params.outputs_prefix} "
        "-p marker_file {output.marker}"


rule all_forecasts:
    input:
        expand(
            DATA_INTERIM_DIR / f"aifs_forecast_{STORM_NAME}_{{init_date}}.done",
            init_date=INIT_DATES,
        ),


rule aifs_tracking:
    input:
        marker=DATA_INTERIM_DIR / f"aifs_forecast_{STORM_NAME}_{{init_date}}.done",
        notebook=NOTEBOOKS_DIR / "04-aifs-tracking.ipynb",
    output:
        msl_nc=DATA_INTERIM_DIR / f"aifs_msl_{STORM_NAME}_{{init_date}}.nc",
        csv=DATA_INTERIM_DIR / f"aifs_tracks_{STORM_NAME}_{{init_date}}.csv",
        notebook=NOTEBOOKS_OUTPUT_DIR
        / f"04-aifs-tracking_{STORM_NAME}_{{init_date}}.ipynb",
    params:
        n_members=AIFS_N_MEMBERS,
        storage_bucket=AIFS_STORAGE_BUCKET,
        outputs_prefix=AIFS_OUTPUTS_PREFIX,
    shell:
        "{PAPERMILL} {input.notebook} {output.notebook} "
        "-p storm_name {STORM_NAME} "
        "-p init_date_str {wildcards.init_date} "
        "-p n_members {params.n_members} "
        "-p storage_bucket {params.storage_bucket} "
        "-p outputs_prefix {params.outputs_prefix} "
        "-p msl_nc_file {output.msl_nc} "
        "-p output_file {output.csv}"


rule all_aifs_tracking:
    input:
        expand(
            DATA_INTERIM_DIR / f"aifs_tracks_{STORM_NAME}_{{init_date}}.csv",
            init_date=INIT_DATES,
        ),


rule track_error:
    input:
        era5_tracks=DATA_INTERIM_DIR / f"era5_tracks_{STORM_NAME}.csv",
        aifs_tracks=expand(
            DATA_INTERIM_DIR / f"aifs_tracks_{STORM_NAME}_{{init_date}}.csv",
            init_date=INIT_DATES,
        ),
        notebook=NOTEBOOKS_DIR / "05-track-error.ipynb",
    output:
        csv=DATA_PROCESSED_DIR / f"track_error_{STORM_NAME}.csv",
        png=DATA_PROCESSED_DIR / f"track_error_{STORM_NAME}.png",
        maps_png=DATA_PROCESSED_DIR / f"track_error_{STORM_NAME}_maps.png",
        notebook=NOTEBOOKS_OUTPUT_DIR / f"05-track-error_{STORM_NAME}.ipynb",
    params:
        interim_dir=DATA_INTERIM_DIR,
        atlantic_lat=ATLANTIC_LAT,
        atlantic_lon=ATLANTIC_LON,
    shell:
        "{PAPERMILL} {input.notebook} {output.notebook} "
        "-p storm_name {STORM_NAME} "
        "-p era5_tracks_file {input.era5_tracks} "
        "-p interim_dir {params.interim_dir} "
        "-p output_csv {output.csv} "
        "-p output_png {output.png} "
        "-p output_maps_png {output.maps_png}"
