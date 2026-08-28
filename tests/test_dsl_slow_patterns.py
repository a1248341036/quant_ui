"""DSL 慢模式静态拦截门禁。

运行 ``scripts/check_dsl_slow_patterns.py``（AST 扫描，不执行代码），断言
``alphaagent/dsl/core/operators.py`` 无**新增**逐品种 pandas 循环 / 纯 Python
逐 bar 循环（存量命中见脚本内 WHITELIST，只减不增）。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT = Path("scripts/check_dsl_slow_patterns.py")


def test_no_new_slow_patterns():
    assert SCRIPT.is_file(), f"拦截脚本缺失: {SCRIPT}"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        "发现新增 DSL 慢模式（逐品种 pandas 循环 / 纯 Python 逐 bar 循环），"
        "请改用 ops_kit.instrument_group_order + accel 边界并行内核模板：\n"
        f"{proc.stdout}\n{proc.stderr}"
    )
