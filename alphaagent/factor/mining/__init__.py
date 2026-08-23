"""因子挖掘插件（可选依赖 mining extra）。"""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "run_factor_mining":
        from alphaagent.factor.mining.run import run_factor_mining

        return run_factor_mining
    if name == "run_factor_mining_agentscope":
        from alphaagent.factor.mining.agentscope_run import run_factor_mining_agentscope

        return run_factor_mining_agentscope
    if name == "MiningConfig":
        from alphaagent.factor.mining.config import MiningConfig

        return MiningConfig
    if name == "build_system_prompt":
        from alphaagent.factor.mining.prompts import build_system_prompt

        return build_system_prompt
    if name == "FactorSubmitService":
        from alphaagent.factor.mining.submit import FactorSubmitService

        return FactorSubmitService
    if name == "default_factorlib_path":
        from alphaagent.factor.mining.submit import default_factorlib_path

        return default_factorlib_path
    if name == "FactorEvalTools":
        from alphaagent.factor.mining.tools import FactorEvalTools

        return FactorEvalTools
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
