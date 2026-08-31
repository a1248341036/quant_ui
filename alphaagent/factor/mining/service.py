"""Shim: actual implementation moved to eval/service.py"""
from alphaagent.factor.mining.eval.service import *  # noqa: F401,F403


def __getattr__(name: str):
    import alphaagent.factor.mining.eval.service as _m
    return getattr(_m, name)
