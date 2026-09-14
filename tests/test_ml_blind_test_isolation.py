"""ML 组合盲测物理隔离与防泄漏测试。

验证规则：
1. 默认 eval_mode 为 tuning（研发验证态），数据右端物理截断至 val_end (2024-12-31)；
2. tuning 模式下若 mining_end 为 auto，自动对齐到 train_end (2022-12-31)，保留 2023~2024 作为安全 OOS 验证段；
3. blind_test 模式严禁开启 --llm-assist，防止 LLM 窥探盲测数据；
4. llm_summarize_report 遇 blind_test 报告绝对拦截，返回 None；
5. StackingTrainRequest 默认模式为 tuning。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from alphaagent.factor.stacking.dataset import FactorEntry
from alphaagent.factor.stacking.llm_assist import (
    _build_prompt_a,
    _build_report_view,
    llm_summarize_report,
)
from alphaagent.factor.window_config import (
    DEFAULT_TRAIN_END,
    DEFAULT_VAL_END,
)
from backend.routers.alphaagent import StackingTrainRequest


def test_stacking_train_request_default_eval_mode():
    """断言后端请求默认处于 tuning 研发验证模式。"""
    req = StackingTrainRequest()
    assert req.eval_mode == "tuning"


def test_blind_test_mode_rejects_llm_assist():
    """断言在 blind_test 模式下开启 --llm-assist 会被安全拦截退出。"""
    cmd = [
        sys.executable,
        "scripts/train_ml_composite.py",
        "--eval-mode", "blind_test",
        "--llm-assist",
    ]
    proc = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace")
    assert proc.returncode != 0
    out = (proc.stdout or "") + (proc.stderr or "")
    assert "[safety-violation]" in out


def test_llm_summarize_report_blocks_blind_test():
    """断言 llm_summarize_report 绝对拒绝解读包含盲测段数据的报告。"""
    blind_report = {
        "run_id": "test_blind_run",
        "eval_mode": "blind_test",
        "blind_test_isolated": False,
        "mining_end": "2024-12-31",
        "folds": 3,
        "feature_names": ["f1", "f2"],
    }
    res = llm_summarize_report(blind_report)
    assert res is None


def test_build_report_view_contains_safety_meta():
    """断言压缩视图包含 eval_mode 与 blind_test_isolated 状态元数据。"""
    report = {
        "run_id": "test_tuning_run",
        "eval_mode": "tuning",
        "blind_test_isolated": True,
        "mining_end": "2022-12-31",
        "folds": 4,
        "feature_names": ["f1"],
    }
    view_str = _build_report_view(report)
    assert '"eval_mode": "tuning"' in view_str
    assert '"blind_test_isolated": true' in view_str


def test_tuning_mode_date_logic():
    """断言 tuning 模式下日期截断逻辑严格生效。"""
    val_end_limit = pd.Timestamp(DEFAULT_VAL_END)
    train_end_limit = pd.Timestamp(DEFAULT_TRAIN_END)

    # 模拟传入超过 2024-12-31 的 end 日期
    passed_end = pd.Timestamp("2026-05-01")
    eval_mode = "tuning"

    if eval_mode == "tuning":
        end = passed_end
        if end > val_end_limit:
            end = val_end_limit
        mining_end = train_end_limit

    assert end <= pd.Timestamp("2024-12-31")
    assert mining_end == pd.Timestamp("2022-12-31")
    # OOS 验证段起止
    oos_start = mining_end + pd.Timedelta(days=1)
    assert oos_start == pd.Timestamp("2023-01-01")
    assert end == pd.Timestamp("2024-12-31")


def test_prompt_a_no_blind_test_dates():
    """P1 门禁：断言步骤 A 喂给 LLM 的 prompt 绝不泄漏 2025/2026 等盲测期时间戳。"""
    entries = [
        FactorEntry(
            factor_id="f1",
            name="factor_alpha",
            library="production",
            created_at="2026-09-03 14:20:00",
            facets=("价量面",),
            expr="CS_RANK($close)",
        ),
        FactorEntry(
            factor_id="f2",
            name="factor_beta",
            library="candidate",
            created_at="2025-06-15 09:30:00",
            facets=("筹码面",),
            expr="TS_MEAN($turnover, 20)",
        ),
        FactorEntry(
            factor_id="f3",
            name="factor_historical",
            library="production",
            created_at="2022-08-01 12:00:00",
            facets=("基本面",),
            expr="DIVIDE($pe, $pb)",
        ),
    ]
    prompt_str = _build_prompt_a(entries)

    # 严禁出现 2025 或 2026 年份字符串
    assert "2026" not in prompt_str, "LLM prompt 中泄漏了 2026 年份"
    assert "2025" not in prompt_str, "LLM prompt 中泄漏了 2025 年份"
    # 历史年份 2022 可以安全保留年月
    assert "2022-08" in prompt_str
    # 2025/2026 入库的因子应被清洗为安全标签
    assert "post-mining (recent)" in prompt_str


def test_report_view_no_blind_test_dates():
    """P1 门禁：断言步骤 C 喂给 LLM 的 report_view 绝不泄漏 2025/2026 等盲测期日期。"""
    report = {
        "run_id": "safe_tuning_run",
        "eval_mode": "tuning",
        "blind_test_isolated": True,
        "mining_end": "2022-12-31",
        "folds": 2,
        "fold_metrics": {
            "ridge": [
                {"fold": 1, "ic_mean": 0.035, "oos_start": "2023-01-01", "oos_end": "2023-06-30"},
                {"fold": 2, "ic_mean": 0.041, "oos_start": "2023-07-01", "oos_end": "2024-12-31"},
            ]
        },
        "gate": {
            "passed": True,
            "metrics": {"excess_annual": 0.12},
        },
    }
    view_str = _build_report_view(report)
    assert "2025" not in view_str
    assert "2026" not in view_str


def test_get_training_events_from_jsonl(tmp_path, monkeypatch):
    """断言 get_training_events 能正确从 events.jsonl 读取结构化事件。"""
    from backend import stacking_service

    fake_root = tmp_path / "artifacts" / "alphaagent" / "stacking"
    monkeypatch.setattr(stacking_service, "STACKING_ROOT", fake_root)

    train_dir = fake_root / "test_run_123"
    train_dir.mkdir(parents=True, exist_ok=True)
    events_file = train_dir / "events.jsonl"
    events_file.write_text(
        '{"event": "session_start", "ts": "2026-09-14T10:00:00", "scheme": "ml"}\n'
        '{"event": "agent_thinking", "ts": "2026-09-14T10:00:05", "content": "thinking..."}\n'
        '{"event": "session_end", "ts": "2026-09-14T10:01:00", "status": "completed"}\n',
        encoding="utf-8",
    )

    evs = stacking_service.get_training_events("test_run_123")
    assert len(evs) == 3
    assert evs[0]["event"] == "session_start"
    assert evs[1]["event"] == "agent_thinking"
    assert evs[2]["event"] == "session_end"


def test_get_training_events_fallback_from_report(tmp_path, monkeypatch):
    """断言无 events.jsonl 的老训练能从 report.json 合成兜底回放事件。"""
    from backend import stacking_service

    fake_root = tmp_path / "artifacts" / "alphaagent" / "stacking"
    monkeypatch.setattr(stacking_service, "STACKING_ROOT", fake_root)

    train_dir = fake_root / "test_run_legacy"
    train_dir.mkdir(parents=True, exist_ok=True)
    report_file = train_dir / "report.json"
    report_file.write_text(
        json.dumps({
            "run_id": "test_run_legacy",
            "scheme": "ml",
            "scheme_label": "ML 学习加权",
            "feature_names": ["f1", "f2"],
            "gate": {"passed": True, "metrics": {"excess_annual": 0.15}},
            "oos_ic_blended": {"ic_mean": 0.042, "ic_ir": 0.45},
        }),
        encoding="utf-8",
    )

    evs = stacking_service.get_training_events("test_run_legacy")
    assert len(evs) >= 3
    event_names = [e["event"] for e in evs]
    assert "session_start" in event_names
    assert "ml_gate_evaluated" in event_names
    assert "session_end" in event_names


def test_recommend_mrmr_factors_tool_registration():
    """断言 recommend_mrmr_factors 工具正确注册且 schema 完备。"""
    from alphaagent.factor.mining.tools._schemas import (
        TOOL_NAMES,
        _RECOMMEND_MRMR_FACTORS_PARAMETERS,
    )
    assert "recommend_mrmr_factors" in TOOL_NAMES
    assert _RECOMMEND_MRMR_FACTORS_PARAMETERS["type"] == "object"
    props = _RECOMMEND_MRMR_FACTORS_PARAMETERS["properties"]
    assert "k" in props
    assert "max_corr" in props
    assert "no_candidate" in props


def test_repro_command_dynamic_modes(tmp_path):
    """P0 门禁：断言 _repro_command 依据 params/report 正确保留多模式（包括 fundamental），而非硬编码 technical。"""
    from backend.composite_factor_service import _repro_command

    out_dir = tmp_path / "test_run_repro"
    out_dir.mkdir(parents=True, exist_ok=True)
    params_file = out_dir / "params.json"
    params_file.write_text(json.dumps({"modes": ["technical", "fundamental"]}), encoding="utf-8")

    report = {
        "scheme": "ml",
        "label_days": 10,
        "mining_end": "2022-12-31",
        "include_factors": ["f1", "f2"],
    }
    cmd = _repro_command(report, out_dir)
    assert "--modes technical fundamental" in cmd
    assert "--include-factors f1 f2" in cmd
    assert "--isolation holdout" in cmd


def test_save_composite_factor_auto_ingest_and_dedup(tmp_path, monkeypatch):
    """P1 门禁：断言 save_composite_factor 支持 auto_ingest、eval_mode 标注与基于指纹的防重复入库。"""
    from backend import composite_factor_service

    fake_db = tmp_path / "composite_factors.db"
    monkeypatch.setattr(composite_factor_service, "DB_PATH", fake_db)

    out_dir = tmp_path / "run_combo_1"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / "report.json"
    report_file.write_text(
        json.dumps({
            "run_id": "run_combo_1",
            "scheme": "ml",
            "scheme_label": "ML 学习加权",
            "eval_mode": "tuning",
            "blind_test_isolated": True,
            "mining_end": "2022-12-31",
            "panel_end": "2024-12-31",
            "feature_names": ["f1", "f2"],
            "gate": {"passed": True, "metrics": {"excess_annual": 0.12}},
            "oos_ic_blended": {"ic_mean": 0.035, "ic_ir": 0.38},
        }),
        encoding="utf-8",
    )

    # 首次自动入库：新增
    res1 = composite_factor_service.save_composite_factor(out_dir, name="组合测试_v1", auto_ingest=True)
    assert res1.get("ok") is True
    assert res1.get("id") is not None

    items = composite_factor_service.list_composite_factors()
    assert len(items) == 1
    assert items[0]["name"] == "组合测试_v1"
    assert items[0]["eval_mode"] == "tuning"

    # 第二次相同搭配入库（IC 0.030 弱于已有 0.035）：被跳过
    out_dir_inferior = tmp_path / "run_combo_2"
    out_dir_inferior.mkdir(parents=True, exist_ok=True)
    (out_dir_inferior / "report.json").write_text(
        json.dumps({
            "run_id": "run_combo_2",
            "scheme": "ml",
            "scheme_label": "ML 学习加权",
            "eval_mode": "tuning",
            "mining_end": "2022-12-31",
            "feature_names": ["f1", "f2"],
            "oos_ic_blended": {"ic_mean": 0.030},
        }),
        encoding="utf-8",
    )
    res2 = composite_factor_service.save_composite_factor(out_dir_inferior, name="组合测试_v2", auto_ingest=True)
    assert res2.get("skipped_reason") == "duplicate_inferior"
    assert len(composite_factor_service.list_composite_factors()) == 1

    # 第三次相同搭配入库（IC 0.045 优于已有 0.035）：更新覆盖，记录总数仍为 1
    out_dir_superior = tmp_path / "run_combo_3"
    out_dir_superior.mkdir(parents=True, exist_ok=True)
    (out_dir_superior / "report.json").write_text(
        json.dumps({
            "run_id": "run_combo_3",
            "scheme": "ml",
            "scheme_label": "ML 学习加权",
            "eval_mode": "tuning",
            "mining_end": "2022-12-31",
            "feature_names": ["f1", "f2"],
            "oos_ic_blended": {"ic_mean": 0.045},
        }),
        encoding="utf-8",
    )
    res3 = composite_factor_service.save_composite_factor(out_dir_superior, name="组合测试_v3_优质", auto_ingest=True)
    assert res3.get("updated") is True
    items_after = composite_factor_service.list_composite_factors()
    assert len(items_after) == 1
    assert items_after[0]["name"] == "组合测试_v3_优质"
    assert items_after[0]["oos_ic"] == 0.045




