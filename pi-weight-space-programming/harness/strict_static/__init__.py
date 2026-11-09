"""Opt-in strict-static weight analysis protocol.

This package is a separate tier from the legacy Python sandbox
(``WSP_WEIGHT_ONLY`` / ``--wsp-mode weight-only``). It does not change
historical audit results.
"""

PROTOCOL_VERSION = "strict-static-1"

__all__ = ["PROTOCOL_VERSION"]
