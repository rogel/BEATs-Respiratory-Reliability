"""Matched batch samplers for multi-database training ablations."""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np
import pandas as pd
from torch.utils.data import Sampler


def _validate_common(
    rows: pd.DataFrame,
    batch_size: int,
    samples_per_epoch: int | None,
) -> tuple[int, int]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    resolved_samples = int(samples_per_epoch or len(rows))
    if resolved_samples < batch_size:
        raise ValueError("samples_per_epoch must allow at least one full batch")
    return resolved_samples, math.ceil(resolved_samples / batch_size)


class EventRandomBatchSampler(Sampler[list[int]]):
    """Sample the raw joint event distribution with matched full-batch steps."""

    def __init__(
        self,
        rows: pd.DataFrame,
        batch_size: int,
        samples_per_epoch: int | None = None,
        seed: int = 20_260_729,
    ) -> None:
        self.batch_size = int(batch_size)
        self.samples_per_epoch, self.num_batches = _validate_common(
            rows,
            self.batch_size,
            samples_per_epoch,
        )
        self.num_rows = len(rows)
        if self.num_rows == 0:
            raise ValueError("rows must not be empty")
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[list[int]]:
        random_generator = np.random.default_rng(self.seed + self.epoch)
        total = self.num_batches * self.batch_size
        complete_repeats, remainder = divmod(total, self.num_rows)
        sampled = [
            random_generator.permutation(self.num_rows)
            for _ in range(complete_repeats)
        ]
        if remainder:
            sampled.append(
                random_generator.choice(
                    self.num_rows,
                    size=remainder,
                    replace=False,
                )
            )
        indices = np.concatenate(sampled)
        random_generator.shuffle(indices)
        for start in range(0, total, self.batch_size):
            yield indices[start : start + self.batch_size].astype(int).tolist()


class DomainClassEventBatchSampler(Sampler[list[int]]):
    """Balance database and class while sampling events uniformly per stratum."""

    def __init__(
        self,
        rows: pd.DataFrame,
        batch_size: int,
        samples_per_epoch: int | None = None,
        seed: int = 20_260_729,
        domain_column: str = "dataset",
        class_column: str = "binary_label_id",
    ) -> None:
        if batch_size < 4 or batch_size % 4:
            raise ValueError("batch_size must be divisible by four")
        required = {domain_column, class_column}
        missing = required.difference(rows.columns)
        if missing:
            raise ValueError(f"rows is missing columns: {sorted(missing)}")
        domains = sorted(str(value) for value in rows[domain_column].unique())
        classes = sorted(int(value) for value in rows[class_column].unique())
        if len(domains) != 2 or classes != [0, 1]:
            raise ValueError("Expected exactly two domains and binary classes")
        self.batch_size = int(batch_size)
        self.per_stratum = self.batch_size // 4
        self.samples_per_epoch, self.num_batches = _validate_common(
            rows,
            self.batch_size,
            samples_per_epoch,
        )
        self.seed = int(seed)
        self.epoch = 0
        self.strata = {
            (domain, class_index): rows.index[
                rows[domain_column].astype(str).eq(domain)
                & rows[class_column].astype(int).eq(class_index)
            ].to_numpy(dtype=np.int64)
            for domain in domains
            for class_index in classes
        }
        empty = [stratum for stratum, indices in self.strata.items() if not len(indices)]
        if empty:
            raise ValueError(f"Empty sampler strata: {empty}")

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[list[int]]:
        random_generator = np.random.default_rng(self.seed + self.epoch)
        for _ in range(len(self)):
            batch = [
                int(index)
                for indices in self.strata.values()
                for index in random_generator.choice(
                    indices,
                    size=self.per_stratum,
                    replace=True,
                )
            ]
            random_generator.shuffle(batch)
            yield batch


class DomainClassPatientBatchSampler(Sampler[list[int]]):
    """Balance domain and class, then sample patients before their events."""

    def __init__(
        self,
        rows: pd.DataFrame,
        batch_size: int,
        samples_per_epoch: int | None = None,
        seed: int = 20_260_729,
        domain_column: str = "dataset",
        class_column: str = "binary_label_id",
        patient_column: str = "patient_id",
    ) -> None:
        if batch_size < 4 or batch_size % 4:
            raise ValueError("batch_size must be divisible by four")
        required = {domain_column, class_column, patient_column}
        missing = required.difference(rows.columns)
        if missing:
            raise ValueError(f"rows is missing columns: {sorted(missing)}")
        domains = sorted(str(value) for value in rows[domain_column].unique())
        classes = sorted(int(value) for value in rows[class_column].unique())
        if len(domains) != 2 or classes != [0, 1]:
            raise ValueError("Expected exactly two domains and binary classes")
        self.batch_size = int(batch_size)
        self.per_stratum = batch_size // 4
        self.samples_per_epoch, self.num_batches = _validate_common(
            rows,
            self.batch_size,
            samples_per_epoch,
        )
        self.seed = int(seed)
        self.epoch = 0
        self.strata: dict[tuple[str, int], dict[str, np.ndarray]] = {}
        for domain in domains:
            for class_index in classes:
                subset = rows.loc[
                    rows[domain_column].astype(str).eq(domain)
                    & rows[class_column].astype(int).eq(class_index)
                ]
                patient_indices = {
                    str(patient): group.index.to_numpy(dtype=np.int64)
                    for patient, group in subset.groupby(patient_column, sort=True)
                }
                if not patient_indices:
                    raise ValueError(f"Empty sampler stratum: {domain}, {class_index}")
                self.strata[(domain, class_index)] = patient_indices

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[list[int]]:
        random_generator = np.random.default_rng(self.seed + self.epoch)
        for _ in range(len(self)):
            batch: list[int] = []
            for patient_indices in self.strata.values():
                patients = np.asarray(tuple(patient_indices))
                sampled_patients = random_generator.choice(
                    patients,
                    size=self.per_stratum,
                    replace=True,
                )
                batch.extend(
                    int(random_generator.choice(patient_indices[str(patient)]))
                    for patient in sampled_patients
                )
            random_generator.shuffle(batch)
            yield batch
