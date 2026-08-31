"""Shim: actual implementation moved to eval/param_stability.py"""
from alphaagent.factor.mining.eval.param_stability import *  # noqa: F401,F403


def __getattr__(name: str):
    import alphaagent.factor.mining.eval.param_stability as _m
    return getattr(_m, name)
