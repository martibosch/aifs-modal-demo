[![GitHub license](https://img.shields.io/github/license/martibosch/aifs-modal-demo.svg)](https://github.com/martibosch/aifs-modal-demo/blob/main/LICENSE)

# AIFS modal demo

Serverless AIFS forecasting in [modal](https://modal.com).

![Example forecast over Europe](figures/forecast-europe.png)

## Requirements

1. A [modal](https://modal.com) account to run serverless AIFS inference. The ["Starter" plan gets you 30$/month of free credits](https://modal.com/pricing), so considering that the running the example 96 h forecast costs about 0.05$, you should be able to run about 600 of them!
2. A [Tigris storage bucket](https://www.tigrisdata.com/docs/buckets/create-bucket) to write the outputs using [Icechunk](https://icechunk.io/en/stable/overview). The ["Free Tier" gives you 5 GB of data storage per month](https://www.tigrisdata.com/pricing), which can be a bit limiting but still should let you run a few forecasts.

## Steps to run

1. Autehticate to your modal account:

```bash
pixi run modal setup
```

2. Set up your Tigris access keys as [a modal secret](https://modal.com/docs/guide/secrets) named `aws-credentials` with at least the `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` keys (and optionally `AWS_REGION`).

3. Run [the `run-aifs-modal` notebook](https://github.com/martibosch/aifs-modal-demo/blob/main/notebooks/run-aifs-modal.ipynb) :rocket:!

## Acknowledgments

- This is an adaptation of the [earth-mover/aifs-demo](https://github.com/earth-mover/aifs-demo) repository to run on [modal](https://modal.com).
- The [aifs-single-v1.1](https://huggingface.co/ecmwf/aifs-single-1.1) model, [Anemoi](https://anemoi.readthedocs.io) framework and [ECMWF Open Data](https://www.ecmwf.int/en/forecasts/datasets/open-data) have been produced by the [European Centre for Medium-Range Weather Forecasts (ECMWF)](https://www.ecmwf.int).
- Based on the [cookiecutter-data-snake :snake:](https://github.com/martibosch/cookiecutter-data-snake) template for reproducible data science.
