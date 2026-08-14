"""Project-specific exception taxonomy."""


class RKLLMProjectError(Exception):
    """Base project error."""


class ConfigurationError(RKLLMProjectError):
    """Raised when project configuration is invalid."""


class ArtifactError(RKLLMProjectError):
    """Raised when a required model artifact is invalid."""


class BackendUnavailableError(RKLLMProjectError):
    """Raised when backend prerequisites are unavailable."""


class NativeRunnerError(RKLLMProjectError):
    """Raised when the native runner cannot serve a request."""
