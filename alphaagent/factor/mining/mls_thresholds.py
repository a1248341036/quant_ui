"""Shim: actual implementation moved to eval/mls_thresholds.py"""
from alphaagent.factor.mining.eval.mls_thresholds import *  # noqa: F401,F403


def __getattr__(name: str):
    import alphaagent.factor.mining.eval.mls_thresholds as _m
    return getattr(_m, name)
