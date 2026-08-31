"""Shim: actual implementation moved to agent/population.py"""
from alphaagent.factor.mining.agent.population import *  # noqa: F401,F403


def __getattr__(name: str):
    import alphaagent.factor.mining.agent.population as _m
    return getattr(_m, name)
