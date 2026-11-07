"""Portable MNIST / CIFAR-10 small-model bank for weight-only backdoor audits.

Operator training code. Checkpoints produced here are not blind-audit inputs
until ``export`` writes a separate blind tree.
"""

PROTOCOL_VERSION = "1"

__version__ = "1.0.0"
