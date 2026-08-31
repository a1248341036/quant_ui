"""Shim: actual implementation moved to eval/response.py"""
from alphaagent.factor.mining.eval.response import *  # noqa: F401,F403


def __getattr__(name: str):
    import alphaagent.factor.mining.eval.response as _m
    return getattr(_m, name)
