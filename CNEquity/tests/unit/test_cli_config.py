from __future__ import annotations

from pathlib import Path

from cnequity.cli.main import resolve_config_path


def test_default_config_falls_back_to_checkout_quant_dataset_config(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    path = resolve_config_path("configs/cnequity.toml")

    assert path.name == "cnequity.quant_dataset.toml"
    assert path.is_file()
    assert Path("configs/cnequity.toml").exists() is False
