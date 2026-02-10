"""AIFS Modal app."""

import contextlib
import datetime
import os
import queue
import threading
from os import path
from typing import TYPE_CHECKING

import modal

LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

# modal app config
GPU_TYPE = "L40S"
DATA_VOLUME_NAME = "aifs-data"
MODELS_VOLUME_NAME = "aifs-models"
DATA_DIR = "/data"
MODELS_DIR = "/models"
APP_NAME = "aifs-modal"

# volumes
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
# volume to store models, i.e., (i) HuggingFace Hub cache, (ii) PyTorch hub cache and
# (iii) our checkpoints (eventually)
models_volume = modal.Volume.from_name(MODELS_VOLUME_NAME, create_if_missing=True)

# secrets
arraylake_api_token_secret = modal.Secret.from_name(
    "arraylake-api-token",  # required_keys=["ARRAYLAKE_API_TOKEN"]
)
aws_credentials_secret = modal.Secret.from_name(
    "aws-credentials",
)

app = modal.App(APP_NAME)

flash_attn_release = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
    "flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
)
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.0-runtime-ubuntu22.04", add_python="3.12"
    )
    .apt_install("git")
    .uv_pip_install(
        # "anemoi-datasets==0.5.21",
        # "anemoi-graphs==0.5.0",
        "anemoi-inference[huggingface]==0.6.3",
        "anemoi-models==0.5.0",
        "anemoi-utils==0.4.22",
        "arraylake",
        "boto3",
        "earthkit-regrid",
        flash_attn_release,
        "icechunk",
        "numpy",
        "torch==2.9.0",
        "torch-geometric==2.4.0",
        "xarray",
        "zarr",
        extra_index_url="https://download.pytorch.org/whl/cu126",
    )
    .env(
        {
            "HF_HUB_CACHE": path.join(MODELS_DIR, "hf_hub_cache"),
            "HF_HUB_ENABLE_HF_TRANSFER": "1",  # turn on faster downloads from HF
            "TORCH_HOME": path.join(MODELS_DIR, "torch"),
            # "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "ANEMOI_INFERENCE_NUM_CHUNKS": "16",
        }
    )
)


if TYPE_CHECKING:
    import icechunk
    import numpy as np


# utils
def datetime_to_str(date: datetime.datetime) -> str:
    """Convert datetime to a string."""
    assert date.tzinfo == datetime.UTC
    assert date.minute == date.second == date.microsecond == 0
    assert date.hour in [0, 6, 12, 18]
    return date.strftime("%Y-%m-%d/%Hz")


@contextlib.contextmanager
def _without_aws_env():
    keys = [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "AWS_PROFILE",
        "AWS_SHARED_CREDENTIALS_FILE",
        "AWS_CONFIG_FILE",
        "AWS_ENDPOINT_URL",
    ]
    saved = {key: os.environ.pop(key) for key in keys if key in os.environ}
    try:
        yield
    finally:
        os.environ.update(saved)


def get_gpu_regridder(source_grid, target_grid, method="linear"):
    """Create a GPU regridder using weights from Earthkit regrid."""
    import earthkit.regrid as ekr
    import torch

    class GPU_Regridder:
        def __init__(self, source_grid, target_grid, method="linear"):
            weights_csr, self.target_shape = ekr.db.find(
                source_grid, target_grid, method
            )
            self.weights = torch.sparse_csr_tensor(
                torch.from_numpy(weights_csr.indptr),
                torch.from_numpy(weights_csr.indices),
                torch.from_numpy(weights_csr.data),
                size=weights_csr.shape,
            ).cuda()

        def regrid(self, data):
            tensor = torch.from_numpy(data.astype("f8")).cuda()
            regridded = self.weights.matmul(tensor)
            return regridded.cpu().numpy().astype("f4").reshape(self.target_shape)

    return GPU_Regridder(source_grid, target_grid, method)


def state_to_xarray(state, regridder, include_pressure_levels=False):
    """Convert the state fields to an xarray dataset."""
    import numpy as np
    import xarray as xr

    fields = state["fields"]
    dims = ("valid_time", "lat", "lon")
    lat = 90 - 0.25 * np.arange(721)
    lon = 0.25 * np.arange(1440)
    pressure = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    ds = xr.Dataset(
        {
            vname: (
                dims,
                regridder.regrid(array)[None, :, :],
            )
            for vname, array in fields.items()
        },
        coords={
            "valid_time": (
                "valid_time",
                [state["date"]],
                {"axis": "T", "standard_name": "time"},
            ),
            "lat": ("lat", lat, {"standard_name": "latitude", "axis": "Y"}),
            "lon": ("lon", lon, {"standard_name": "longitude", "axis": "X"}),
            "pressure": pressure,
        },
    )
    ds.valid_time.encoding.update(
        {"units": "hours since 1970-01-01T00:00:00", "chunks": (1200,)}
    )

    to_drop = []
    for pvar in ["q", "t", "u", "v", "w", "z"]:
        vnames = [f"{pvar}_{plev}" for plev in pressure]
        if include_pressure_levels:
            ds[pvar] = xr.concat(
                [ds[vname] for vname in vnames], dim="pressure"
            ).transpose("valid_time", ...)
        to_drop.extend(vnames)

    ds = ds.drop_vars(to_drop)

    return ds


def fetch_initial_conditions(
    date: datetime.datetime, session: "icechunk.Session"
) -> dict[str, "np.ndarray"]:
    """Fetch initial conditions for a given date."""
    import numpy as np
    import zarr

    group_prev = zarr.open_group(
        session.store,
        zarr_format=3,
        path=datetime_to_str(date - datetime.timedelta(hours=6)),
        mode="r",
    )
    group_curr = zarr.open_group(
        session.store, zarr_format=3, path=datetime_to_str(date), mode="r"
    )

    vnames_curr = group_curr["variable"][:]
    vnames_prev = group_prev["variable"][:]
    np.testing.assert_equal(vnames_curr, vnames_prev)

    fields_prev = group_prev["fields"][:]
    fields_curr = group_curr["fields"][:]
    data = np.stack([fields_prev, fields_curr], axis=1)

    # tweak data to conform with AIFS input format
    mapping = {"sot_1": "stl1", "sot_2": "stl2", "vsw_1": "swvl1", "vsw_2": "swvl2"}

    def maybe_rename_vname(vname):
        if vname in mapping:
            return mapping[vname]
        return vname

    fields = {
        maybe_rename_vname(vnames_curr[n]): data[n] for n in range(len(vnames_curr))
    }

    # convert to geopotential height
    for level in LEVELS:
        gh = fields.pop(f"gh_{level}")
        fields[f"z_{level}"] = gh * 9.80665

    return fields


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=60 * 60 * 4,
    volumes={DATA_DIR: data_volume},
    secrets=[arraylake_api_token_secret, aws_credentials_secret],
)
def run_forecast(
    date: datetime.datetime,
    # target_storage_path: str,
    target_storage_bucket: str,
    *,
    lead_time: int = 96,
    initial_conditions_repo: str = "earthmover-public/aifs-initial-conditions",
    initial_conditions_branch: str = "main",
    target_storage_prefix: str = "aifs-outputs",
    target_branch: str = "main",
    checkpoint: dict | None = None,
) -> None:  # dict[str, str]:
    """Run forecast."""
    import arraylake as al
    import icechunk
    import torch
    from anemoi.inference.outputs.printer import print_state
    from anemoi.inference.runners.simple import SimpleRunner

    date_no_tz = date.replace(tzinfo=None)

    # load initial conditions from arraylake
    with _without_aws_env():
        client = al.Client(token=os.environ["ARRAYLAKE_API_TOKEN"])
        # client.login()
        # ACHTUNG: use default config to avoid confusion with AWS credentials
        config = icechunk.RepositoryConfig.default()
        repo = client.get_repo(initial_conditions_repo, config=config)
        input_session = repo.readonly_session(initial_conditions_branch)

    # date = datetime.datetime(2025, 9, 15, 6, 0, 0, tzinfo=datetime.UTC)
    print("loading initial conditions for", date)
    # fields = load_fields(session, input_group)
    fields = fetch_initial_conditions(date, input_session)
    input_state = dict(date=date_no_tz, fields=fields)

    # # prepare output storage on a modal volume
    target_storage = icechunk.tigris_storage(
        bucket=target_storage_bucket,
        prefix=target_storage_prefix,
        region=os.getenv("AWS_REGION", None),
        access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        # endpoint_url=os.environ["AWS_ENDPOINT_URL"],
        # from_env=True,
    )
    target_repo = icechunk.Repository.open_or_create(target_storage)
    target_session = target_repo.writable_session(target_branch)

    if checkpoint is None:
        checkpoint = {"huggingface": "ecmwf/aifs-single-1.1"}

    runner = SimpleRunner(checkpoint, device="cuda")

    q = queue.Queue()
    lock = threading.Lock()

    def worker():
        while True:
            (ds, store, group_name, kwargs) = q.get()
            with lock:
                ds.to_zarr(
                    store, group=group_name, zarr_format=3, consolidated=False, **kwargs
                )
            q.task_done()

    threading.Thread(target=worker, daemon=True).start()

    torch.cuda.empty_cache()

    regridder = get_gpu_regridder({"grid": "N320"}, {"grid": (0.25, 0.25)})

    print("starting forecast loop")
    kwargs = {"mode": "w"}

    for n, state in enumerate(runner.run(input_state=input_state, lead_time=lead_time)):
        print_state(state)
        ds = state_to_xarray(state, regridder=regridder).chunk()
        group = datetime_to_str(date)
        if n > 0:
            kwargs = {"mode": "a", "append_dim": "valid_time"}
        q.put((ds, target_session.store, group, kwargs))

    # wait for all I/O tasks to finish
    q.join()

    torch.cuda.empty_cache()
    target_session.commit(
        f"{lead_time} hour forecast for {date.strftime('%Y-%m-%d %H:%M')} written to "
        f"{target_storage_bucket}"
    )
