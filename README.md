# BEATs Respiratory Reliability

Python tools and verification data for respiratory-sound classification, audio-model adaptation and predictive reliability.

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
| `data/manifests/` | Event identifiers, labels and data-role assignments |
| `artifacts/` | Predictions, statistical outputs, diagnostics and resource measurements |
| `runs/` | Saved validation predictions, configurations and execution records |

## Verify the supplied results

See [Data and result guide](DATA_GUIDE.md) for the file index, schema, comparison definitions and verification scope. The read-only checks require Python with NumPy, pandas and SciPy, and do not need source recordings, model weights or training:

```sh
python -m pip install numpy pandas scipy
python -B scripts/verify_reference_results.py
python -B scripts/verify_ablation_results.py
```

`DATA_MANIFEST.json` records SHA-256 checksums for every supplied data/result file. The ablation verifier checks these hashes before independently reconstructing the metrics. Run the commands from the repository root. Preserve the supplied files and use separate output paths for new experiments.

## Getting started

Obtain the datasets and pretrained weights separately, following [Data sources and licences](DATA_SOURCES_AND_LICENCES.md). The repository includes derived predictions, data-role assignments and statistical outputs. Source recordings, original annotation files and model weights are obtained from the original providers and are not redistributed.

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
  --output-dir outputs/manifests
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
