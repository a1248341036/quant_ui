"""Shim: actual implementation moved to eval/env_settings.py"""
from alphaagent.factor.mining.eval.env_settings import *  # noqa: F401,F403


def __getattr__(name: str):
    import alphaagent.factor.mining.eval.env_settings as _m
    return getattr(_m, name)
