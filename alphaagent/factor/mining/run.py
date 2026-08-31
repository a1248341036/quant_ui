"""Shim: actual implementation moved to agent/run.py"""
from alphaagent.factor.mining.agent.run import *  # noqa: F401,F403


def __getattr__(name: str):
    import alphaagent.factor.mining.agent.run as _m
    return getattr(_m, name)
