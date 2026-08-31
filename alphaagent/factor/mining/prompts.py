"""Shim: actual implementation moved to prompt/prompts.py"""
from alphaagent.factor.mining.prompt.prompts import *  # noqa: F401,F403


def __getattr__(name: str):
    import alphaagent.factor.mining.prompt.prompts as _m
    return getattr(_m, name)
