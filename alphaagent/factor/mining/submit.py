"""Shim: actual implementation moved to delivery/submit.py"""
from alphaagent.factor.mining.delivery.submit import *  # noqa: F401,F403


def __getattr__(name: str):
    import alphaagent.factor.mining.delivery.submit as _m
    return getattr(_m, name)
