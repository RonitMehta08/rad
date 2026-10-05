"""Structured exception types for RadQueue AI (MASTER_PROMPT §19.4).

Owner: P1 (Data & Config Lead); used by all layers.
"""

from __future__ import annotations


class RadQueueError(Exception):
    """Base class for all RadQueue AI errors."""


class ModelNotAvailableError(RadQueueError):
    """A trained model artifact is missing or could not be loaded."""


class ConfigurationError(RadQueueError):
    """A configuration file is missing or invalid."""


class InvalidInputError(RadQueueError):
    """User/API input failed domain validation."""


class ResourceNotFoundError(RadQueueError):
    """A referenced machine, patient or scenario does not exist."""
