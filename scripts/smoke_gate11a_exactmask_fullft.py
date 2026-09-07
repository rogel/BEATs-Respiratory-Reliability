#!/usr/bin/env python3
"""Run pre-freeze Gate 11A quality checks without validation_select access."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import torch
import yaml
from compute_feature_stats import feature_config_from_yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader
from train_gate11a_exactmask_fullft import (
    MemoryTracker,
    _augmentation,
    _classifier_parameters,
    _cpu_clone,
    _model_state_cpu,
    _sampler_state,
    _train_one_epoch,
    _validate_sampler_state,
)

from respiratory_sound.data.audio import ICBHICycleDataset
from respiratory_sound.data.sampling import DomainClassEventBatchSampler
from respiratory_sound.gate11a import (
    ALLOWED_SEEDS,
    TRAIN_ROLE,
    assert_selected_rows,
    sha256_file,
)
from respiratory_sound.models.pretrained_audio import (
    _beats_token_valid_mask,
    load_beats_transfer,
)
from respiratory_sound.runtime import select_device
from respiratory_sound.training import (
    consistency_weight,
    seed_everything,
    warmup_cosine_scheduler,
)
from respiratory_sound.training_state import (
    atomic_torch_save,
    capture_rng_state,
    restore_rng_state,
)


class TinyWaveformModel(nn.Module):
    """Small deterministic model for the epoch-boundary recovery audit."""

    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(3, 2)

    def forward(self, waveforms: Tensor, sample_masks: Tensor) -> Tensor:
        weights = sample_masks.to(dtype=waveforms.dtype)
        counts = weights.sum(dim=1).clamp_min(1.0)
        mean = (waveforms * weights).sum(dim=1) / counts
        centered = (waveforms - mean[:, None]) * weights
        standard_deviation = (
            centered.square().sum(dim=1) / counts
        ).sqrt()
        absolute_mean = (waveforms.abs() * weights).sum(dim=1) / counts
        features = torch.stack([mean, standard_deviation, absolute_mean], dim=1)
        return self.classifier(features)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/cross_domain_binary.csv"),
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=Path("configs/data/gate9a_beats.yaml"),
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/model/gate9a_beats_as2m_cpt2.yaml"),
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=Path(
            "configs/experiment/"
            "gate11a_beats_exactmask_fullft_seed20260729.yaml"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/gate11a_exactmask_fullft_quality_smoke.json"
        ),
    )
    parser.add_argument("--device", choices=("mps",), default="mps")
    return parser.parse_args()


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _assert_no_gate11a_results(root: Path) -> None:
    result_names = {
        "history.csv",
        "summary.json",
        "decision.json",
        "best.pt",
        "resume_state.pt",
    }
    violations = [
        str(path.relative_to(root))
        for run_dir in (root / "runs").glob("gate11a_exactmask_fullft_seed*")
        for path in run_dir.iterdir()
        if path.name in result_names
        or path.name.startswith("best_validation_predictions_")
    ]
    if violations:
        raise RuntimeError(
            f"Formal Gate 11A results exist before quality smoke: {violations}"
        )


def _train_dataset(
    *,
    root: Path,
    manifest_path: Path,
    data_config_path: Path,
    experiment: dict[str, Any],
) -> ICBHICycleDataset:
    dataset = ICBHICycleDataset(
        manifest_path=manifest_path,
        project_root=root,
        split_column="protocol_role",
        split_value=TRAIN_ROLE,
        feature_config=feature_config_from_yaml(data_config_path),
        training=True,
        num_views=2,
        augmentation=_augmentation(experiment),
        label_column=str(experiment["label_column"]),
        return_waveform=True,
        waveform_only=True,
    )
    assert_selected_rows(dataset.rows, expected_role=TRAIN_ROLE)
    return dataset


def _one_batch_loader(
    dataset: ICBHICycleDataset,
    *,
    seed: int,
    generator: torch.Generator,
) -> tuple[DomainClassEventBatchSampler, DataLoader]:
    sampler = DomainClassEventBatchSampler(
        dataset.rows,
        batch_size=8,
        samples_per_epoch=8,
        seed=seed,
        class_column="binary_label_id",
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=0,
        pin_memory=False,
        generator=generator,
    )
    return sampler, loader


def _recorded_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    consistency_strength: float,
    label_lookup: dict[str, int],
) -> tuple[dict[str, float], list[str]]:
    sample_ids: list[str] = []

    def batches() -> Any:
        for batch in loader:
            sample_ids.extend(str(value) for value in batch[3])
            yield batch

    metrics = _train_one_epoch(
        model,
        batches(),
        optimizer,
        device,
        consistency_strength=consistency_strength,
        memory_tracker=MemoryTracker(),
        label_lookup=label_lookup,
    )
    return metrics, sample_ids


def _tiny_recovery_audit(
    dataset: ICBHICycleDataset,
    experiment: dict[str, Any],
) -> dict[str, Any]:
    seed = ALLOWED_SEEDS[0]
    label_lookup = dict(
        zip(
            dataset.rows["sample_id"].astype(str),
            dataset.rows["binary_label_id"].astype(int),
            strict=True,
        )
    )

    def construct() -> tuple[
        TinyWaveformModel,
        torch.optim.Optimizer,
        Any,
        torch.Generator,
        DomainClassEventBatchSampler,
        DataLoader,
    ]:
        model = TinyWaveformModel()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1.0e-3,
            weight_decay=0.01,
        )
        scheduler = warmup_cosine_scheduler(
            optimizer,
            warmup_epochs=3,
            max_epochs=20,
        )
        generator = torch.Generator()
        generator.manual_seed(seed + 11_000_000)
        sampler, loader = _one_batch_loader(
            dataset,
            seed=seed,
            generator=generator,
        )
        return model, optimizer, scheduler, generator, sampler, loader

    seed_everything(seed)
    model, optimizer, scheduler, generator, sampler, loader = construct()
    sampler.set_epoch(0)
    first_metrics, first_ids = _recorded_epoch(
        model,
        loader,
        optimizer,
        torch.device("cpu"),
        consistency_strength=consistency_weight(
            0,
            maximum=float(experiment["consistency"]["max_weight"]),
            warmup_epochs=int(experiment["consistency"]["warmup_epochs"]),
        ),
        label_lookup=label_lookup,
    )
    scheduler.step()
    boundary_state = {
        "schema_version": 1,
        "next_epoch": 1,
        "model_state_dict": _model_state_cpu(model),
        "optimizer_state_dict": _cpu_clone(optimizer.state_dict()),
        "scheduler_state_dict": scheduler.state_dict(),
        "rng_state": capture_rng_state(torch.device("cpu")),
        "sampler_state": _sampler_state(sampler, completed_epoch=0),
        "train_loader_generator_state": generator.get_state(),
    }
    with tempfile.TemporaryDirectory(prefix="gate11a-recovery-") as directory:
        state_path = Path(directory) / "resume_state.pt"
        atomic_torch_save(boundary_state, state_path)
        saved = torch.load(state_path, map_location="cpu", weights_only=False)

    sampler.set_epoch(1)
    uninterrupted_metrics, uninterrupted_ids = _recorded_epoch(
        model,
        loader,
        optimizer,
        torch.device("cpu"),
        consistency_strength=consistency_weight(
            1,
            maximum=float(experiment["consistency"]["max_weight"]),
            warmup_epochs=int(experiment["consistency"]["warmup_epochs"]),
        ),
        label_lookup=label_lookup,
    )
    uninterrupted_state = _model_state_cpu(model)

    seed_everything(seed)
    (
        resumed_model,
        resumed_optimizer,
        resumed_scheduler,
        resumed_generator,
        resumed_sampler,
        resumed_loader,
    ) = construct()
    resumed_model.load_state_dict(saved["model_state_dict"], strict=True)
    resumed_optimizer.load_state_dict(saved["optimizer_state_dict"])
    resumed_scheduler.load_state_dict(saved["scheduler_state_dict"])
    _validate_sampler_state(
        saved["sampler_state"],
        resumed_sampler,
        next_epoch=1,
    )
    resumed_generator.set_state(saved["train_loader_generator_state"])
    restore_rng_state(saved["rng_state"], torch.device("cpu"))
    resumed_sampler.set_epoch(1)
    resumed_metrics, resumed_ids = _recorded_epoch(
        resumed_model,
        resumed_loader,
        resumed_optimizer,
        torch.device("cpu"),
        consistency_strength=consistency_weight(
            1,
            maximum=float(experiment["consistency"]["max_weight"]),
            warmup_epochs=int(experiment["consistency"]["warmup_epochs"]),
        ),
        label_lookup=label_lookup,
    )
    resumed_state = _model_state_cpu(resumed_model)
    metrics_exact = uninterrupted_metrics == resumed_metrics
    sample_ids_exact = uninterrupted_ids == resumed_ids
    parameters_exact = all(
        torch.equal(uninterrupted_state[name], resumed_state[name])
        for name in uninterrupted_state
    )
    if not (metrics_exact and sample_ids_exact and parameters_exact):
        raise RuntimeError("Gate 11A epoch-boundary recovery is not exact")
    return {
        "first_epoch_metrics": first_metrics,
        "first_epoch_sample_ids": first_ids,
        "next_epoch_metrics_exact": metrics_exact,
        "next_epoch_sample_ids_exact": sample_ids_exact,
        "next_epoch_parameters_exact": parameters_exact,
        "next_epoch_sample_ids": resumed_ids,
    }


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    manifest_path = _resolve(root, args.manifest).resolve()
    data_config_path = _resolve(root, args.data_config).resolve()
    model_config_path = _resolve(root, args.model_config).resolve()
    experiment_config_path = _resolve(root, args.experiment_config).resolve()
    output_path = _resolve(root, args.output).resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite quality evidence: {output_path}")
    _assert_no_gate11a_results(root)

    experiment = yaml.safe_load(
        experiment_config_path.read_text(encoding="utf-8")
    )
    model_config = yaml.safe_load(
        model_config_path.read_text(encoding="utf-8")
    )
    seed = ALLOWED_SEEDS[0]
    seed_everything(seed)
    device = select_device(args.device)
    if device.type != "mps":
        raise RuntimeError("Gate 11A quality smoke requires MPS")

    dataset = _train_dataset(
        root=root,
        manifest_path=manifest_path,
        data_config_path=data_config_path,
        experiment=experiment,
    )
    label_lookup = dict(
        zip(
            dataset.rows["sample_id"].astype(str),
            dataset.rows["binary_label_id"].astype(int),
            strict=True,
        )
    )
    generator = torch.Generator()
    generator.manual_seed(seed + 11_000_000)
    sampler, loader = _one_batch_loader(
        dataset,
        seed=seed,
        generator=generator,
    )
    sampler.set_epoch(0)
    batch = next(iter(loader))

    checkpoint_path = (root / str(model_config["checkpoint"])).resolve()
    source_dir = (root / str(model_config["source_dir"])).resolve()
    model = load_beats_transfer(checkpoint_path, source_dir)
    backbone_parameters, classifier_parameters = _classifier_parameters(model)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if total_parameters != trainable_parameters:
        raise RuntimeError("Quality smoke did not configure full finetuning")
    model = model.to(device)

    waveforms, sample_masks, _, _ = batch
    flat_waveforms = waveforms.reshape(-1, waveforms.shape[-1])
    flat_masks = sample_masks.reshape(-1, sample_masks.shape[-1])
    fbanks = model.backbone.preprocess(flat_waveforms.to("cpu"))
    patch_size = int(model.backbone.input_patch_size)
    frequency_patches = int(fbanks.shape[2] // patch_size)
    token_valid = _beats_token_valid_mask(
        flat_masks.to("cpu"),
        fbank_frames=int(fbanks.shape[1]),
        patch_size=patch_size,
        frequency_patches=frequency_patches,
    )
    valid_token_counts = token_valid.sum(dim=1)
    if bool((valid_token_counts <= 0).any()) or bool(
        (valid_token_counts > 400).any()
    ):
        raise RuntimeError("Exact token mask count is outside [1, 400]")

    optimizer = torch.optim.AdamW(
        [
            {
                "params": backbone_parameters,
                "lr": float(
                    experiment["optimizer"]["backbone_learning_rate"]
                ),
            },
            {
                "params": classifier_parameters,
                "lr": float(experiment["optimizer"]["head_learning_rate"]),
            },
        ],
        weight_decay=float(experiment["optimizer"]["weight_decay"]),
    )
    memory_tracker = MemoryTracker()
    full_ft_metrics = _train_one_epoch(
        model,
        [batch],
        optimizer,
        device,
        consistency_strength=consistency_weight(
            0,
            maximum=float(experiment["consistency"]["max_weight"]),
            warmup_epochs=int(experiment["consistency"]["warmup_epochs"]),
        ),
        memory_tracker=memory_tracker,
        label_lookup=label_lookup,
    )
    torch.mps.synchronize()
    memory_tracker.update(device)
    del optimizer
    del model
    torch.mps.empty_cache()

    recovery_audit = _tiny_recovery_audit(dataset, experiment)
    payload = {
        "schema_version": 1,
        "gate": "11A",
        "status": "pre_freeze_quality_smoke_passed",
        "formal_validation_accessed": False,
        "selected_role": TRAIN_ROLE,
        "selected_samples": len(dataset),
        "actual_beats_fullft_step": {
            "passed": True,
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "trainable_fraction": trainable_parameters / total_parameters,
            "metrics": full_ft_metrics,
            "memory": memory_tracker.to_dict(),
        },
        "exact_token_mask": {
            "passed": True,
            "fbank_frames": int(fbanks.shape[1]),
            "patch_size": patch_size,
            "frequency_patches": frequency_patches,
            "total_tokens": int(token_valid.shape[1]),
            "valid_token_count_min": int(valid_token_counts.min()),
            "valid_token_count_max": int(valid_token_counts.max()),
            "all_counts_finite_and_nonzero": True,
        },
        "epoch_boundary_recovery": recovery_audit,
        "inputs": {
            "manifest_sha256": sha256_file(manifest_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "data_config_sha256": sha256_file(data_config_path),
            "model_config_sha256": sha256_file(model_config_path),
            "experiment_config_sha256": sha256_file(experiment_config_path),
        },
        "calibration_accessed": False,
        "locked_tests_accessed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(payload, output_path)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
