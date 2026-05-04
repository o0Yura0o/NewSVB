# -*- coding: utf-8 -*-
"""
NSVB-ZH binarizer.

Key differences from the original PopBuTFyENSpkEMBinarizer:
    - Two independent datasets (M4Singer + OpenSinger), NO amateur/pro pairing.
    - Each item gets `dataset_label` in {0, 1} (0 = M4, 1 = OpenSinger).
    - Phoneme dict is Chinese (use_tone=true).
    - No SADTW, no a2p alignment, no cross_item linking.

Directory conventions expected (override via hparams):
    raw_data_dir/
        m4singer/<singer>/<song>/<utt>.wav      (+ optional .txt lyrics)
        opensinger/<split>/<singer>/<utt>.wav   (+ optional .txt)
"""
import os
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np

from data_gen.singing.binarize import SingingBinarizer          # base class
from utils.hparams import hparams


class NsvbZhBinarizer(SingingBinarizer):
    """Binarise M4Singer and OpenSinger into a single combined DB with labels.

    The task side reads `dataset_label` per item and routes batches at
    Stage 2. Batches for Stage 1 ignore the label (or feed it to the GRL
    domain head).
    """

    DATASET_ROOTS = {
        0: 'm4singer',
        1: 'opensinger',
    }

    # ---------------------------------------------------------------------
    # Item discovery
    # ---------------------------------------------------------------------
    def meta_data(self) -> Iterator[Tuple[str, str, int]]:
        """Yield (item_name, wav_path, dataset_label)."""
        raw_root = Path(hparams['raw_data_dir'])
        label_map = hparams.get('dataset_labels', {
            'm4singer': 0, 'opensinger': 1,
        })
        for name, label in label_map.items():
            ds_root = raw_root / name
            if not ds_root.exists():
                print(f"[binarize] WARN: {ds_root} does not exist, skipping.")
                continue
            for wav_path in sorted(ds_root.rglob('*.wav')):
                rel = wav_path.relative_to(raw_root)
                item_name = str(rel).replace('/', '__').replace('\\', '__').rsplit('.', 1)[0]
                yield item_name, str(wav_path), label

    # ---------------------------------------------------------------------
    # Per-item processing
    # ---------------------------------------------------------------------
    def process_item(self, item_name: str, wav_path: str,
                     dataset_label: int, tg_path: str = None) -> dict:
        """Extract mel, F0, phonemes, and attach dataset_label.

        Falls back on the base class for mel+F0; adds txt→phonemes if a
        matching .txt file is present next to the wav.
        """
        item = super().process_item(item_name, wav_path, tg_path=tg_path)
        if item is None:
            return None
        item['dataset_label'] = int(dataset_label)

        txt_path = Path(wav_path).with_suffix('.txt')
        if txt_path.exists():
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    item['txt'] = f.read().strip()
            except Exception as e:
                print(f"[binarize] WARN: could not read {txt_path}: {e}")

        return item

    # ---------------------------------------------------------------------
    # Split
    # ---------------------------------------------------------------------
    def split_train_test_valid(self, items):
        """Stratified split so each dataset is present in train/val/test."""
        items_by_label = {0: [], 1: []}
        for it in items:
            items_by_label[it['dataset_label']].append(it)

        train, valid, test = [], [], []
        n_valid = hparams.get('valid_set_size', 200)
        n_test = hparams.get('test_set_size', 200)
        for lbl, bucket in items_by_label.items():
            rng = np.random.default_rng(seed=hparams.get('seed', 42) + lbl)
            rng.shuffle(bucket)
            half_v = n_valid // 2
            half_t = n_test // 2
            valid.extend(bucket[:half_v])
            test.extend(bucket[half_v:half_v + half_t])
            train.extend(bucket[half_v + half_t:])
        return train, valid, test
