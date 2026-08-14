from pathlib import Path

from rk_llm.types import SystemMetrics


def collect_system_metrics(
    meminfo_path: Path = Path("/proc/meminfo"),
    temperature_path: Path = Path("/sys/class/thermal/thermal_zone0/temp"),
) -> SystemMetrics:
    memory_mb: float | None = None
    temperature_c: float | None = None
    if meminfo_path.is_file():
        for line in meminfo_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                memory_mb = float(line.split()[1]) / 1024.0
                break
    if temperature_path.is_file():
        temperature_c = (
            float(temperature_path.read_text(encoding="utf-8").strip()) / 1000.0
        )
    return SystemMetrics(memory_mb, temperature_c, None, None)
