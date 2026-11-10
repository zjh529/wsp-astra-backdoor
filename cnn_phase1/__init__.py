"""Phase-1 study: can gpt-6-astra find and undo a CNN backdoor from weights alone?

Training, the blind audit, and scoring are separate entry points. The audit
path does not import the scorer and does not read ground truth.
"""

__version__ = "0.1.0"
