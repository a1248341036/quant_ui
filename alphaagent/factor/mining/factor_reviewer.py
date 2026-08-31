"""Shim: actual implementation moved to agent/factor_reviewer.py"""
from alphaagent.factor.mining.agent.factor_reviewer import *  # noqa: F401,F403


def __getattr__(name: str):
    import alphaagent.factor.mining.agent.factor_reviewer as _m
    return getattr(_m, name)
