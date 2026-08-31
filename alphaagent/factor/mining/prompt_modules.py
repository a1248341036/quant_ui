"""Shim: actual implementation moved to prompt/prompt_modules.py"""
from alphaagent.factor.mining.prompt.prompt_modules import *  # noqa: F401,F403


def __getattr__(name: str):
    import alphaagent.factor.mining.prompt.prompt_modules as _m
    return getattr(_m, name)
