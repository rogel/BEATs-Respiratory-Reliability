# BEATs Respiratory Reliability

Python tools for respiratory-sound classification, audio-model adaptation and predictive reliability.

## Features

- Waveform preprocessing, augmentation and position-aware propagation of valid audio support into BEATs tokens.
- Full fine-tuning, low-rank adaptation (LoRA) and other audio-model adaptation strategies.
- Patient-disjoint data preparation and database/class-balanced sampling.
- Probability calibration, threshold analysis, selective prediction and class-specific coverage checks.
- Training, checkpoint evaluation, paired statistical comparisons and resource profiling.

## Installation

Use Python 3.11 or 3.12 in an isolated environment:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Dependencies are declared in `pyproject.toml`. `constraints-lock.txt` records a reference environment. PyTorch and torchaudio must be compatible with each other and with the selected device. Training command-line tools support CPU and Apple MPS where indicated by their `--help` output.

## Project layout

| Directory | Contents |
| --- | --- |
| `src/respiratory_sound/` | Data processing, models, adaptation, training, calibration and evaluation |
| `configs/` | Data, model, training and ablation configurations |
| `scripts/` | Command-line preparation, training, evaluation and analysis tools |
| `tests/` | Unit and synthetic-data regression tests |
| `third_party/beats/` | Vendored BEATs runtime code and its original licence |

## Getting started

Obtain the datasets and pretrained weights separately, following [Data sources and licences](DATA_SOURCES_AND_LICENCES.md). This repository contains software and configuration files; datasets, trained weights, predictions and run outputs are not bundled.

Inspect the available data preparation and training arguments:

```sh
python scripts/prepare_icbhi.py --help
python scripts/prepare_sprsound.py --help
python scripts/train_icbhi.py --help
```

After obtaining the ICBHI source files, generate a manifest:

```sh
python scripts/prepare_icbhi.py \
  --audio-dir /path/to/ICBHI_final_database \
  --split-file /path/to/ICBHI_challenge_train_test.txt \
  --output-dir data/manifests
```

Review the selected data/model/training configuration before starting a run. Some experiment drivers use fixed seeds, checksums, saved manifests, checkpoint identities or external plan files; they require those inputs and are not one-command demonstrations. Keep their integrity guards enabled, and use a separate output directory for new experiments. Scripts bearing `gate` or `revision` names are experiment-stage utilities, not generic CLI entry points.

## Tests

Run after installing the development dependencies:

```sh
python -m pytest tests -q
```

The unit and synthetic-data tests do not train a full model or establish performance on a real dataset.

## Licensing

See [LICENSING.md](LICENSING.md). Third-party copyright notices and licence terms are retained in their original files.
