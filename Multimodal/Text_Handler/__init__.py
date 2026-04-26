"""
Text Handler package.

Exports preprocessing components used by the ImageBind pipeline.
"""

from .preprocessing import AudioClassifier, TextProducer, run_preprocessing

__all__ = [
	"AudioClassifier",
	"TextProducer",
	"run_preprocessing",
]
