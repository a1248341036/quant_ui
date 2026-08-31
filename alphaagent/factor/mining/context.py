"""Shim: actual implementation moved to eval/context.py"""
from alphaagent.factor.mining.eval.context import *  # noqa: F401,F403


def __getattr__(name: str):
    import alphaagent.factor.mining.eval.context as _m
    return getattr(_m, name)
