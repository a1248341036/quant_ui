"""Shim: actual implementation moved to delivery/delivery_criteria.py"""
from alphaagent.factor.mining.delivery.delivery_criteria import *  # noqa: F401,F403


def __getattr__(name: str):
    import alphaagent.factor.mining.delivery.delivery_criteria as _m
    return getattr(_m, name)
