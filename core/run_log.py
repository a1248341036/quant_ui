"""把 quant_ui 脚本的运行记录写入 CNE 的 manifest.db。

复用 CNEquity.orchestrator.manifest.Manifest 的 start_run / finish_run，
让 ETF/基金/一键刷新等脚本的执行历史统一出现在 CNE runs 页面
（http://127.0.0.1:8787/#/runs）。

设计约定
--------
* job_name 统一加 ``quant_ui:`` 前缀，与 CNE 自身的 run 区分。
* manifest.db 路径来自 CNE 配置（data_root/meta/manifest.db），
  与 cne_reader 共享同一套懒加载逻辑。
* **静默降级**：CNE 不可用时只打印 warning，不影响脚本正常执行。
  run 记录是"锦上添花"，不能让日志系统拖垮数据刷新。
* 提供上下文管理器 ``job(...)`` 自动处理 start/finish/异常。

用法
-----
.. code-block:: python

    from core.run_log import job

    with job("quant_ui:refresh_etf", metadata={"mode": "incremental"}) as run:
        # ... 刷新逻辑 ...
        run.set_rows(rows_written=len(etf_panel))
        # 如果出错，上下文管理器自动 finish_run(status="failed")

手动调用::

    from core.run_log import start_run, finish_run
    run_id = start_run("quant_ui:refresh_etf")
    try:
        ...
        finish_run(run_id, status="success", rows_written=1000)
    except Exception as exc:
        finish_run(run_id, status="failed", error_message=str(exc))
        raise
"""
from __future__ import annotations

import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# CNE manifest.db 路径（与 cne_reader 同源）
_CNE_ROOT = Path(__file__).resolve().parent.parent / "CNEquity"
_CNE_CONFIG = _CNE_ROOT / "configs" / "cnequity.quant_dataset.toml"

_manifest: Any = None
_manifest_path: Path | None = None
_init_error: str | None = None


def _resolve_manifest() -> Any:
    """懒加载 Manifest 实例；CNE 不可用时返回 None。

    直接用 importlib 从文件路径加载 manifest.py 模块，绕过
    cnequity.__init__ 的链式 import（会拉起 polars/pyarrow）。
    Manifest 类本身只用 sqlite3 + 标准库，无第三方依赖。
    """
    global _manifest, _manifest_path, _init_error
    if _manifest is not None:
        return _manifest
    if _init_error is not None:
        return None
    try:
        import importlib.util

        manifest_py = _CNE_ROOT / "src" / "cnequity" / "orchestrator" / "manifest.py"
        if not manifest_py.is_file():
            raise FileNotFoundError(f"manifest.py not found: {manifest_py}")

        # 直接从文件加载模块，跳过 cnequity/__init__.py 的链式 import
        spec = importlib.util.spec_from_file_location("_cne_manifest", manifest_py)
        mod = importlib.util.module_from_spec(spec)
        # dataclass 装饰器需要 sys.modules 里有该模块名
        sys.modules["_cne_manifest"] = mod
        spec.loader.exec_module(mod)
        Manifest = mod.Manifest

        # manifest.db 路径 = CNEquity/data/quant_dataset/_cnequity/meta/manifest.db
        # CNE 配置 [data] root = "data/quant_dataset/_cnequity" 是相对 CNEquity/ 目录
        # 解析的（cne serve 从 CNEquity/ 目录启动），所以真实路径在 CNEquity 下。
        _manifest_path = (
            _CNE_ROOT / "data" / "quant_dataset" / "_cnequity" / "meta" / "manifest.db"
        )
        if not _manifest_path.exists():
            # 备选：项目根目录下的同名路径
            alt = (
                Path(__file__).resolve().parent.parent
                / "data" / "quant_dataset" / "_cnequity" / "meta" / "manifest.db"
            )
            if alt.exists():
                _manifest_path = alt
        _manifest = Manifest(_manifest_path)
        logger.info("run_log: manifest.db @ %s", _manifest_path)
        return _manifest
    except Exception as exc:  # noqa: BLE001
        _init_error = str(exc)
        logger.warning("run_log: manifest.db 初始化失败，运行记录将不写入: %s", exc)
        return None


def get_manifest() -> Any:
    """获取 Manifest 实例（可能为 None）。"""
    return _resolve_manifest()


def start_run(job_name: str, metadata: dict[str, Any] | None = None) -> str | None:
    """开始一个 run，返回 run_id；CNE 不可用时返回 None。"""
    m = _resolve_manifest()
    if m is None:
        return None
    try:
        return m.start_run(job_name, metadata=metadata)
    except Exception as exc:  # noqa: BLE001
        logger.warning("run_log: start_run 失败: %s", exc)
        return None


def finish_run(
    run_id: str | None,
    status: str,
    rows_read: int = 0,
    rows_written: int = 0,
    error_message: str | None = None,
) -> None:
    """结束一个 run；run_id 为 None 时直接跳过。"""
    if run_id is None:
        return
    m = _resolve_manifest()
    if m is None:
        return
    try:
        m.finish_run(
            run_id,
            status=status,
            rows_read=rows_read,
            rows_written=rows_written,
            error_message=error_message,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("run_log: finish_run 失败: %s", exc)


class RunContext:
    """上下文管理器：自动 start_run + finish_run。"""

    def __init__(self, job_name: str, metadata: dict[str, Any] | None = None):
        self._job_name = job_name
        self._metadata = metadata
        self._run_id: str | None = None
        self._rows_read = 0
        self._rows_written = 0
        self._error: str | None = None

    @property
    def run_id(self) -> str | None:
        return self._run_id

    def set_rows(self, rows_read: int = 0, rows_written: int = 0) -> None:
        self._rows_read = rows_read
        self._rows_written = rows_written

    def __enter__(self) -> "RunContext":
        self._run_id = start_run(self._job_name, self._metadata)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is not None:
            error_message = f"{exc_val}\n{traceback.format_exc()}" if exc_val else str(exc_type)
            finish_run(
                self._run_id,
                status="failed",
                rows_read=self._rows_read,
                rows_written=self._rows_written,
                error_message=error_message[:2000] if error_message else None,
            )
        else:
            finish_run(
                self._run_id,
                status="success",
                rows_read=self._rows_read,
                rows_written=self._rows_written,
            )
        # 不吞异常
        return False


def job(job_name: str, metadata: dict[str, Any] | None = None) -> RunContext:
    """便捷工厂：``with job("quant_ui:refresh_etf") as run:``"""
    return RunContext(job_name, metadata)
