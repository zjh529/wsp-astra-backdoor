"""Anonymous static audit of PADBench LoRA adapters.

The model under test writes ordinary Python. This package runs that Python in a
bubblewrap sandbox and records the session. It does not implement a backdoor
detector.
"""

__version__ = "0.1.0"
