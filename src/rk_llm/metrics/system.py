import math
from pathlib import Path

from rk_llm.types import SystemMetrics


def _memory_available_mb(path: Path) -> float | None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            fields = line.split()
            if not fields or fields[0] != "MemAvailable:":
                continue
            if len(fields) != 3 or fields[2] != "kB":
                return None
            memory_kb = float(fields[1])
            if not math.isfinite(memory_kb) or memory_kb < 0:
                return None
            return memory_kb / 1024.0
    except (OSError, UnicodeError, ValueError):
        return None
    return None


def _temperature_c(path: Path) -> float | None:
    try:
        temperature_c = float(path.read_text(encoding="utf-8").strip()) / 1000.0
    except (OSError, UnicodeError, ValueError):
        return None
    return temperature_c if math.isfinite(temperature_c) else None


def collect_system_metrics(
    meminfo_path: Path = Path("/proc/meminfo"),
    temperature_path: Path = Path("/sys/class/thermal/thermal_zone0/temp"),
) -> SystemMetrics:
    return SystemMetrics(
        _memory_available_mb(meminfo_path),
        _temperature_c(temperature_path),
        None,
        None,
    )
