"""Domain types for text generation and backend reporting."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 1
    repeat_penalty: float = 1.1

    def __post_init__(self) -> None:
        if not self.prompt.strip():
            raise ValueError("prompt must not be empty")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be greater than 0 and at most 1")
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")
        if self.repeat_penalty <= 0:
            raise ValueError("repeat_penalty must be greater than 0")


@dataclass(frozen=True)
class TextChunk:
    text: str
    token_count: int = 1


@dataclass(frozen=True)
class BackendCapabilities:
    name: str
    available: bool
    streaming: bool
    target: str
    is_mock: bool
    reason: str | None = None


@dataclass(frozen=True)
class GenerationResult:
    text: str
    generated_tokens: int
    backend: str
    is_mock: bool
    ttft_ms: float
    decode_ms: float
    tokens_per_second: float
    termination_reason: str = "completed"


@dataclass(frozen=True)
class BenchmarkRecord:
    prompt: str
    iteration: int
    result: GenerationResult


@dataclass(frozen=True)
class SystemMetrics:
    memory_available_mb: float | None
    temperature_c: float | None
    cpu_percent: float | None
    npu_percent: float | None
