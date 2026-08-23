from __future__ import annotations

from datetime import date

from cnequity.config import Config
from cnequity.steps.external import step_local_assets_daily


def test_local_assets_step_runs_both_refreshers(monkeypatch, tmp_path):
    calls = []

    class Completed:
        returncode = 0

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return Completed()

    monkeypatch.setattr("cnequity.steps.external.subprocess.run", fake_run)
    cfg = Config(data_root=tmp_path, external_local_assets_enabled=True)

    result = step_local_assets_daily(cfg, date(2026, 8, 20), "run-1", {})

    assert result.get("status", "success") == "success"
    assert len(calls) == 2
    assert calls[0][0][-2:] == ["--end", "2026-08-20"]
    assert calls[1][0][-2:] == ["2026-08-20", "--no-fees"]


def test_local_assets_step_skips_when_disabled(tmp_path):
    result = step_local_assets_daily(Config(data_root=tmp_path), date(2026, 8, 20), "run-1", {})

    assert result["status"] == "warning"
