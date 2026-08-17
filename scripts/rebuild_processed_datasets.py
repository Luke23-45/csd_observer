"""Rebuild all processed datasets from raw files into the new split layout."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

# Add src to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

import yaml  # noqa: E402

from csd_observer.datasets.provision import provision_dataset  # noqa: E402
from csd_observer.datasets.registry import get_datamodule, get_dataset  # noqa: E402


def rebuild_all() -> None:
    data_root = root_dir / "datasets"
    processed_dir = data_root / "processed"

    print("=" * 60)
    print("REBUILDING PROCESSED DATASETS")
    print(f"Data Root: {data_root}")
    print("=" * 60)

    # 1. Clean existing processed directory
    if processed_dir.exists():
        print(f"Cleaning existing processed directory: {processed_dir}")
        shutil.rmtree(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # 2. Rebuild TAC dataset
    tac_config_path = root_dir / "configs" / "dataset" / "tac.yaml"
    with open(tac_config_path, encoding="utf-8") as f:
        tac_config = yaml.safe_load(f)

    print("\n--- Provisioning TAC dataset ---")
    tac_state = provision_dataset("tac", tac_config, root=data_root, force_rebuild=True)
    print(f"TAC Provision State: {tac_state}")

    # Verify TAC splits & DataModule
    tac_dm = get_datamodule("tac", overrides={"data_root": str(data_root)})
    tac_dm.setup()
    print("TAC Splits loaded:", list(tac_dm._splits.keys()))
    for s_name, s_data in tac_dm._splits.items():
        print(f"  Split [{s_name}]: features shape={s_data['features'].shape}, seq_lengths={s_data['seq_lengths'].shape}, signal count={s_data['is_positive'].sum()}/{len(s_data['is_positive'])}")

    train_loader = tac_dm.train_dataloader()
    batch_x, batch_l, batch_y = next(iter(train_loader))
    print(f"  TAC Train Batch: x={batch_x.shape}, lens={batch_l.shape}, y={batch_y.shape}")

    # 3. Rebuild DaphniaExt dataset
    daphnia_config_path = root_dir / "configs" / "dataset" / "daphnia_ext.yaml"
    with open(daphnia_config_path, encoding="utf-8") as f:
        daphnia_config = yaml.safe_load(f)

    print("\n--- Provisioning DaphniaExt dataset ---")
    daphnia_state = provision_dataset("daphnia_ext", daphnia_config, root=data_root, force_rebuild=True)
    print(f"DaphniaExt Provision State: {daphnia_state}")

    # Verify Daphnia splits & DataModule
    daphnia_dm = get_datamodule("daphnia_ext", overrides={"data_root": str(data_root)})
    daphnia_dm.setup()
    print("Daphnia Splits loaded:", list(daphnia_dm._splits.keys()))
    for s_name, s_data in daphnia_dm._splits.items():
        print(f"  Split [{s_name}]: features shape={s_data['features'].shape}, seq_lengths={s_data['seq_lengths'].shape}, signal count={s_data['is_positive'].sum()}/{len(s_data['is_positive'])}")

    daphnia_train_loader = daphnia_dm.train_dataloader()
    d_batch_x, d_batch_l, d_batch_y = next(iter(daphnia_train_loader))
    print(f"  Daphnia Train Batch: x={d_batch_x.shape}, lens={d_batch_l.shape}, y={d_batch_y.shape}")

    # 4. Verify Synthetic DataModule
    print("\n--- Testing Synthetic Fold DataModule ---")
    syn_dm = get_datamodule("synthetic_fold", overrides={"n_trajectories": 16, "max_length": 64})
    syn_dm.setup()
    print("Synthetic Fold Splits loaded:", list(syn_dm._splits.keys()))
    for s_name, s_data in syn_dm._splits.items():
        print(f"  Split [{s_name}]: features shape={s_data['features'].shape}")

    # 5. Verify get_dataset() compatibility
    print("\n--- Testing get_dataset() compatibility ---")
    tac_bundle = get_dataset("tac", overrides={"data_root": str(data_root)})
    print(f"get_dataset('tac') -> signal features: {tac_bundle['signal']['features'].shape}, null features: {tac_bundle['null']['features'].shape}")

    daphnia_bundle = get_dataset("daphnia_ext", overrides={"data_root": str(data_root)})
    print(f"get_dataset('daphnia_ext') -> signal features: {daphnia_bundle['signal']['features'].shape}, null features: {daphnia_bundle['null']['features'].shape}")

    print("\n" + "=" * 60)
    print("ALL PROCESSED DATASETS REBUILT AND VERIFIED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    rebuild_all()
