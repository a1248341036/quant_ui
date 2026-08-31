"""Shim: actual implementation moved to eval/session.py"""
from alphaagent.factor.mining.eval.session import *  # noqa: F401,F403


def __getattr__(name: str):
    import alphaagent.factor.mining.eval.session as _m
    return getattr(_m, name)
