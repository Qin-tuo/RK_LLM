"""Project-specific exception taxonomy."""


class ProjectError(Exception):
    """Base project error."""


class ConfigurationError(ProjectError):
    """Raised when project configuration is invalid."""


class ArtifactError(ProjectError):
    """Raised when a required model artifact is invalid."""


class BackendUnavailableError(ProjectError):
    """Raised when backend prerequisites are unavailable."""


class NativeRunnerError(ProjectError):
    """Raised when the native runner cannot serve a request."""
