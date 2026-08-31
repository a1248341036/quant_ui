"""Shim: actual implementation moved to eval/schemas.py"""
from alphaagent.factor.mining.eval.schemas import *  # noqa: F401,F403


def __getattr__(name: str):
    import alphaagent.factor.mining.eval.schemas as _m
    return getattr(_m, name)
