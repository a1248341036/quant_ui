"""Shim: actual implementation moved to prompt/operators.py"""
from alphaagent.factor.mining.prompt.operators import *  # noqa: F401,F403


def __getattr__(name: str):
    import alphaagent.factor.mining.prompt.operators as _m
    return getattr(_m, name)
