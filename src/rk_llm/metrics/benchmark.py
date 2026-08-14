import json
from dataclasses import asdict
from pathlib import Path

from rk_llm.backends.base import GenerationBackend
from rk_llm.config import BenchmarkConfig
from rk_llm.generation.service import GenerationService
from rk_llm.types import BenchmarkRecord, GenerationRequest


def run_benchmark(
    config: BenchmarkConfig, backend: GenerationBackend, output: Path
) -> list[BenchmarkRecord]:
    output.parent.mkdir(parents=True, exist_ok=True)
    service = GenerationService(backend)
    records: list[BenchmarkRecord] = []
    for prompt in config.prompts:
        for iteration in range(1, config.iterations + 1):
            request = GenerationRequest(
                prompt=prompt,
                max_new_tokens=config.runtime.max_new_tokens,
                temperature=config.runtime.temperature,
                top_p=config.runtime.top_p,
                top_k=config.runtime.top_k,
                repeat_penalty=config.runtime.repeat_penalty,
            )
            records.append(BenchmarkRecord(prompt, iteration, service.generate(request)))
    output.write_text(
        "".join(
            json.dumps(asdict(record), ensure_ascii=False) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    return records
