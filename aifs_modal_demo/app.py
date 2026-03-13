"""AIFS Modal app."""

import contextlib
import datetime
import os
import queue
import threading
from os import path

import icechunk
import modal
import zarr

from aifs_modal_demo import ingest, utils

# AIFS checkpoints
AIFS_SINGLE_CHECKPOINT = {"huggingface": "ecmwf/aifs-single-1.1"}
AIFS_ENS_CHECKPOINT = {"huggingface": "ecmwf/aifs-ens-1.0"}

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
# arraylake is optional: only included when ARRAYLAKE_API_TOKEN is set locally,
# i.e., when the user wants to pull initial conditions from an Arraylake repo
_secrets = [modal.Secret.from_name("aws-credentials")]
if os.getenv("ARRAYLAKE_API_TOKEN"):
    _secrets.append(modal.Secret.from_name("arraylake-api-token"))
# hf token is also optional (enable faster downloads)
if os.getenv("HF_HUB_TOKEN"):
    _secrets.append(modal.Secret.from_name("huggingface-secret"))

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
        "anemoi-models==0.5.1",
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
            "TORCH_HOME": path.join(MODELS_DIR, "torch"),
            # "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "ANEMOI_INFERENCE_NUM_CHUNKS": "16",
        }
    )
)


# utils
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


# app
@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=60 * 60 * 4,
    volumes={DATA_DIR: data_volume, MODELS_DIR: models_volume},
    secrets=_secrets,
)
def run_forecast(
    date: datetime.datetime,
    # target_storage_path: str,
    storage_bucket: str,
    *,
    lead_time: int = 96,
    initial_conditions_repo: str | None = None,
    initial_conditions_prefix: str | None = None,
    initial_conditions_branch: str = "main",
    outputs_prefix: str = "aifs-outputs",
    outputs_branch: str = "main",
    checkpoint: dict | None = None,
    n_members: int | None = None,
    include_pressure_levels: bool = False,
) -> None:  # dict[str, str]:
    """Run forecast."""
    import arraylake as al
    import torch
    import xarray as xr
    from anemoi.inference.outputs.printer import print_state
    from anemoi.inference.runners.simple import SimpleRunner

    date_no_tz = date.replace(tzinfo=None)

    # get initial conditions session
    if initial_conditions_repo is not None:
        # if a repo is passed, it takes precedence
        # load initial conditions from arraylake
        with _without_aws_env():
            # we need the context manager to avoid arraylake issues with aws env vars
            client = al.Client(token=os.environ["ARRAYLAKE_API_TOKEN"])
            # client.login()
            # ACHTUNG: use default config to avoid confusion with AWS credentials
            config = icechunk.RepositoryConfig.default()
            initial_conditions_session = client.get_repo(
                initial_conditions_repo, config=config
            ).readonly_session(initial_conditions_branch)
    else:
        # get initial conditions from bucket
        initial_conditions_storage = icechunk.tigris_storage(
            bucket=storage_bucket,
            prefix=initial_conditions_prefix,
            region=os.getenv("AWS_REGION", None),
            access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        )
        initial_conditions_session = icechunk.Repository.open(
            initial_conditions_storage
        ).writable_session(initial_conditions_branch)

    # get outputs session
    outputs_storage = icechunk.tigris_storage(
        bucket=storage_bucket,
        prefix=outputs_prefix,
        region=os.getenv("AWS_REGION", None),
        access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )
    outputs_repo = icechunk.Repository.open_or_create(outputs_storage)
    if outputs_branch not in outputs_repo.list_branches():
        base = outputs_repo.readonly_session("main").snapshot_id
        outputs_repo.create_branch(outputs_branch, base)

    # check if requested forecasts already exist
    # (if a target forecast already exists, do not re-run it)
    base_group = utils.datetime_to_str(date)
    readonly_session = outputs_repo.readonly_session(outputs_branch)

    if n_members is not None:
        # ensemble: check if group already has all members
        try:
            existing = xr.open_dataset(
                readonly_session.store,
                group=base_group,
                engine="zarr",
                zarr_format=3,
                chunks=None,
            )
            if existing.sizes.get("ensemble_member", 0) >= n_members:
                print(
                    f"Ensemble forecast already complete for {date.isoformat()} "
                    f"({n_members} members); skipping"
                )
                return
        except (zarr.errors.GroupNotFoundError, Exception):
            pass
    else:
        # deterministic: check if group exists
        try:
            zarr.open_group(
                readonly_session.store, path=base_group, mode="r", zarr_format=3
            )
            print(
                f"Forecast already exists for {date.isoformat()} "
                f"(group: {base_group}); skipping"
            )
            return
        except zarr.errors.GroupNotFoundError:
            pass

    outputs_session = outputs_repo.writable_session(outputs_branch)

    print("loading initial conditions for", date)
    fields = ingest.fetch_initial_conditions(date, initial_conditions_session)
    input_state = dict(date=date_no_tz, fields=fields)

    if checkpoint is None:
        if n_members is not None:
            checkpoint = AIFS_ENS_CHECKPOINT
        else:
            checkpoint = AIFS_SINGLE_CHECKPOINT

    runner = SimpleRunner(checkpoint, device="cuda")

    # filter input fields to only include variables the checkpoint expects
    expected_vars = set(runner.checkpoint.variable_to_input_tensor_index)
    extra = set(fields) - expected_vars
    if extra:
        print(f"dropping {len(extra)} variables not expected by checkpoint: {extra}")
        fields = {k: v for k, v in fields.items() if k in expected_vars}
        input_state = dict(date=date_no_tz, fields=fields)
    missing = expected_vars - set(fields)
    if missing:
        # computed forcings (e.g. cos_latitude, insolation) are injected by the
        # runner automatically — just log them for visibility
        print(f"{len(missing)} variables will be computed by the runner: {missing}")

    # prepare regridder
    regridder = get_gpu_regridder({"grid": "N320"}, {"grid": (0.25, 0.25)})

    # run forecasts
    if n_members is not None:
        # ensemble mode
        # Accumulate all lead times per member, then write each member with an
        # "ensemble_member" dimension.  Member 0 creates the group; subsequent members
        # append along "ensemble_member".
        for m in range(n_members):
            torch.manual_seed(m)
            torch.cuda.empty_cache()
            print(f"running ensemble member {m}/{n_members - 1}")

            steps = []
            for state in runner.run(input_state=input_state, lead_time=lead_time):
                print_state(state)
                steps.append(
                    state_to_xarray(
                        state,
                        regridder=regridder,
                        include_pressure_levels=include_pressure_levels,
                        # init_time=date_no_tz,
                    )
                )

            member_ds = xr.concat(steps, dim="valid_time")
            member_ds = member_ds.expand_dims(ensemble_member=[m]).chunk()

            if m == 0:
                member_ds.to_zarr(
                    outputs_session.store,
                    group=base_group,
                    zarr_format=3,
                    consolidated=False,
                    mode="w",
                )
            else:
                member_ds.to_zarr(
                    outputs_session.store,
                    group=base_group,
                    zarr_format=3,
                    consolidated=False,
                    append_dim="ensemble_member",
                )
            del steps, member_ds
    else:
        # single (deterministic) mode
        # stream each lead time to zarr via a background writer thread
        q = queue.Queue()
        lock = threading.Lock()

        def worker():
            while True:
                (ds, store, group_name, kwargs) = q.get()
                with lock:
                    ds.to_zarr(
                        store,
                        group=group_name,
                        zarr_format=3,
                        consolidated=False,
                        **kwargs,
                    )
                q.task_done()

        threading.Thread(target=worker, daemon=True).start()

        kwargs = {"mode": "w"}
        for n, state in enumerate(
            runner.run(input_state=input_state, lead_time=lead_time)
        ):
            print_state(state)
            ds = state_to_xarray(
                state,
                regridder=regridder,
                include_pressure_levels=include_pressure_levels,
            ).chunk()
            group = utils.datetime_to_str(date)
            if n > 0:
                kwargs = {"mode": "a", "append_dim": "valid_time"}
            q.put((ds, outputs_session.store, group, kwargs))

        # wait for all I/O tasks to finish
        q.join()

    torch.cuda.empty_cache()
    member_str = f" ({n_members} members)" if n_members else ""
    commit_msg = (
        f"{lead_time} hour forecast{member_str} for "
        f"{date.strftime('%Y-%m-%d %H:%M')} written to {outputs_repo}"
    )
    outputs_session.commit(commit_msg)
    print(commit_msg)
