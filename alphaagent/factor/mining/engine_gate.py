"""Shim: actual implementation moved to delivery/engine_gate.py"""
from alphaagent.factor.mining.delivery.engine_gate import *  # noqa: F401,F403


def __getattr__(name: str):
    import alphaagent.factor.mining.delivery.engine_gate as _m
    return getattr(_m, name)
