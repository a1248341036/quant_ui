"""盲测段隔离回归测试（2026-09-24，分支 fix/blind-segment-isolation）。

背景
----
``StockEvalContext.coverage_range()`` 让会话 panel 覆盖 **train ∪ val ∪ test**，
盲测隔离依赖下游按 split 切片。修复前三处正交/相似度计算直接消费全区间数据，
把盲测段（2025-01-01 起）算出的结论回流给了 LLM：

1. ``_orthogonality_check``（agentscope_tools）—— 随机锚点约 25% 落在 test 段；
2. ``_candidate_registry_similarity``（submit）—— 全量逐日，含 test 段；
3. ``SimilarityMatrix.cross_sectional_neighbor_report``（zoo）—— 抽样行
   14.3%（85678/100000）落在 test 段，且它是 stage_one 的准入判决依据。

对应修复见 ``docs/specs/alphaagent_mine_precheck_spec.md`` §0。
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

from alphaagent.factor.mining.eval.context import StockEvalContext

VISIBLE_START = "2020-01-01"
VISIBLE_END = "2024-12-31"
TEST_START = "2025-01-01"


def _mk_panel(start: str, end: str, n_inst: int = 2) -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="B")
    inst = [f"{i:06d}" for i in range(n_inst)]
    idx = pd.MultiIndex.from_product([dates, inst], names=["datetime", "instrument"])
    return pd.DataFrame({"close": np.arange(len(idx), dtype=float)}, index=idx)


class _Ctx:
    train_start = VISIBLE_START
    val_end = VISIBLE_END
    test_start = TEST_START

    def visible_range(self) -> tuple[str, str]:
        return (self.train_start, self.val_end)


class _Session:
    def __init__(self, panel: pd.DataFrame) -> None:
        self.ctx = _Ctx()
        self.panel = panel


# ── 1. visible_range 的语义 ──────────────────────────────────────────────


def test_visible_range_excludes_test_segment() -> None:
    ctx = StockEvalContext(panel_path="cne://")
    vis = ctx.visible_range()
    cov = ctx.coverage_range()

    assert vis == (ctx.train_start, ctx.val_end)
    assert pd.Timestamp(vis[1]) < pd.Timestamp(ctx.test_start)
    # coverage 必须覆盖到 test（交付终审需要）——这正是隔离不能依赖 panel 边界的原因
    assert pd.Timestamp(cov[1]) >= pd.Timestamp(ctx.test_start)


# ── 2. zoo 抽样行的日期掩码 ─────────────────────────────────────────────


class _FakeIndex:
    def __init__(self) -> None:
        self.rows = pd.DataFrame(
            {
                "row_id": np.arange(6),
                "datetime": pd.to_datetime(
                    [
                        "2024-12-30",
                        "2024-12-31",
                        "2025-01-02",
                        "2025-01-03",
                        "2026-01-05",
                        "2026-01-06",
                    ]
                ),
            }
        )
        self.sample_row_ids = np.array([0, 2, 4])


class _FakeZoo:
    def __init__(self) -> None:
        self.index = _FakeIndex()


def test_sample_date_mask_drops_blind_rows() -> None:
    from alphaagent.factor.zoo.similarity import _sample_date_mask

    mask = _sample_date_mask(_FakeZoo(), VISIBLE_END)
    assert mask is not None
    # 抽样行 row_id 0/2/4 → 2024-12-30 / 2025-01-02 / 2026-01-05
    assert mask.tolist() == [True, False, False]


def test_sample_date_mask_returns_none_when_unavailable() -> None:
    from alphaagent.factor.zoo.similarity import _sample_date_mask

    class _Broken:
        @property
        def index(self):  # noqa: ANN201
            raise RuntimeError("boom")

    # 不可得时返回 None（调用方保持原口径），而不是抛异常打断判决
    assert _sample_date_mask(_Broken(), VISIBLE_END) is None


# ── 3. 正交召回：面板与采样锚点都不得触达盲测段 ──────────────────────────


def test_visible_panel_and_sampling_never_touch_test() -> None:
    from alphaagent.factor.mining.agent.agentscope_tools import (
        _sample_orthogonality_panel,
        _visible_panel,
    )

    panel = _mk_panel("2020-01-01", "2026-09-24")
    session = _Session(panel)

    visible = _visible_panel(session)
    assert visible is not None
    assert visible.index.get_level_values("datetime").max() <= pd.Timestamp(VISIBLE_END)
    assert len(visible) < len(panel)

    sampled = _sample_orthogonality_panel(visible)
    sdates = pd.DatetimeIndex(sampled.index.get_level_values("datetime").unique())
    assert len(sdates) > 0
    assert sdates.max() <= pd.Timestamp(VISIBLE_END), "正交采样泄漏进盲测段"


def test_visible_panel_none_without_ctx() -> None:
    from alphaagent.factor.mining.agent.agentscope_tools import _visible_panel

    class _NoCtx:
        panel = pd.DataFrame({"close": [1.0]})

    # 区间不可知 → 返回 None，调用方跳过检查（宁缺勿泄漏）
    assert _visible_panel(_NoCtx()) is None


# ── 4. 候选池正交：registry 的 DSL 必须在收窄后的 panel 上求值 ───────────


def test_candidate_registry_similarity_evaluates_on_visible_range(tmp_path, monkeypatch) -> None:
    import alphaagent.dsl as dsl
    from alphaagent.factor.mining.delivery import submit as sub

    reg = tmp_path / "reg.json"
    reg.write_text(
        json.dumps({"f_old": {"expr": "CS_RANK($close)", "name": "old"}}),
        encoding="utf-8",
    )
    panel = _mk_panel("2020-01-01", "2026-09-24")
    seen: dict[str, object] = {}

    def _fake_eval(expr, p):  # noqa: ANN001, ANN202
        seen["max_dt"] = p.index.get_level_values("datetime").max()
        seen["n_rows"] = len(p)
        raise RuntimeError("recorded")

    # 函数内 `from alphaagent.dsl import eval_factor` → patch 模块属性即可生效
    monkeypatch.setattr(dsl, "eval_factor", _fake_eval)

    sub._candidate_registry_similarity(
        np.arange(len(panel), dtype=float),
        panel,
        reg,
        visible_range=(VISIBLE_START, VISIBLE_END),
    )

    assert seen["max_dt"] <= pd.Timestamp(VISIBLE_END)
    assert seen["n_rows"] < len(panel)


# ── 5. 防回退契约：挖掘链路的调用点必须传隔离参数 ────────────────────────


def test_mining_callers_pass_isolation_params() -> None:
    from alphaagent.factor import ingest as ing
    from alphaagent.factor.mining.delivery import submit as sub

    submit_src = pathlib.Path(sub.__file__).read_text(encoding="utf-8")
    assert "date_max=ctx.val_end" in submit_src, "stage_one 相似度必须剔除盲测段"

    ingest_src = pathlib.Path(ing.__file__).read_text(encoding="utf-8")
    assert "date_max=pol.val_end" in ingest_src, "入库判重相似度必须剔除盲测段"


# ── 6. 研究记忆注入：盲测段指标不得进 LLM 上下文 ────────────────────────


def test_memory_entry_render_excludes_blind_metrics() -> None:
    """``_format_entry`` 无过滤渲染 metrics，而 research memory 里存了 test 段指标。

    该渲染结果经 ``context_for`` **每轮注入** LLM，等同于把样本外 IC 交给模型选因子。
    存储层保留（UI 三段表要展示），注入层必须剔除。
    """
    from alphaagent.factor.mining.memory.retrieval import (
        _BLIND_METRIC_KEYS,
        RetrievalMixin,
    )

    entry = {
        "verdict": "candidate_approved",
        "factor_name": "demo_factor",
        "conclusion": "通过海选进入候选池。",
        "metrics": {
            "ic": 0.03,
            "icir": 0.35,
            "train_ic": 0.031,
            "val_ic": 0.028,
            "test_ic": 0.021,
            "test_icir": 0.22,
            "test_rank_ic": 0.019,
            "test_ic_retention": 0.52,
        },
    }
    text = RetrievalMixin._format_entry(entry, include_expression=False)

    # 训练/验证段指标照常注入
    assert "ic=0.03" in text
    assert "val_ic=0.028" in text
    # 盲测段指标一律不得出现
    for key in _BLIND_METRIC_KEYS:
        assert key not in text, f"盲测指标 {key} 泄漏进注入文本：{text}"
