# FedODD Federated Training

This repository contains the basic training code for FedODD.

## Installation

```bash
pip install -e .
```

## Training

Run the default federated training experiment with:

```bash
flwr run .
```

The main training entry points are `fl/server_app.py` and `fl/client_app.py`. Basic experiment settings, such as the number of clients, training rounds, learning rate, and local epochs, can be changed in `pyproject.toml`.

The active dataset is configured in `fl/task.py` through `CURRENT_DATASET` and `DATASET_CONFIGS`.

Training outputs are saved under `outputs/`.

A detailed README file with instructions will be provided upon acceptance of the paper.
