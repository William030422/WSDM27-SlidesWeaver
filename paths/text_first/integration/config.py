from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


_SIZE_PATTERN = re.compile(r"^\s*(\d+)\s*[xX\u00d7]\s*(\d+)\s*$")
DECK_FONT = "Noto Sans"
DEFAULT_DIFFUSION_MODEL_PATH = Path.home() / "models" / "Tongyi-MAI_Z-Image-Turbo"


@dataclass(frozen=True)
class Size:
    width: int
    height: int

    def as_text(self) -> str:
        return f"{self.width}x{self.height}"

    def as_list(self) -> list[int]:
        return [self.width, self.height]


@dataclass(frozen=True)
class PipelineConfig:
    project_root: Path
    env_path: Path
    api_root: str
    api_key_env: str
    chat_model: str
    image_model: str
    image_size: Size
    image_resolution: str
    image_quality: str
    image_max_attempts: int
    chat_max_tokens: int
    vlm_max_tokens: int
    timeout_seconds: float
    poll_initial_delay_seconds: float
    poll_interval_seconds: float
    poll_timeout_seconds: float
    diffusion_model_path: Path
    diffusion_size: Size
    diffusion_steps: int
    diffusion_guidance_scale: float
    diffusion_seed: int
    deck_font: str


@lru_cache(maxsize=None)
def validate_deck_font(font_family: str = DECK_FONT) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for style in ("Regular", "Bold"):
        result = subprocess.run(
            [
                "fc-match",
                "-f",
                "%{family}\n%{style}\n%{file}\n",
                f"{font_family}:style={style}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        lines = result.stdout.strip().splitlines()
        if result.returncode != 0 or len(lines) < 3:
            raise RuntimeError(f"cannot resolve deck font {font_family} {style}")
        families = {item.strip() for item in lines[0].split(",")}
        resolved_style = lines[1].casefold()
        font_path = Path(lines[2])
        exact_style = (
            "bold" in resolved_style
            if style == "Bold"
            else any(name in resolved_style for name in ("regular", "book", "roman"))
        )
        if font_family not in families or not exact_style or not font_path.is_file():
            raise RuntimeError(
                f"deck font substitution is forbidden: requested {font_family} {style}, "
                f"resolved {lines[0]} {lines[1]}"
            )
        resolved[style.casefold()] = str(font_path.resolve())
    return resolved


def parse_16_9_size(value: str, *, field_name: str) -> Size:
    match = _SIZE_PATTERN.fullmatch(value or "")
    if match is None:
        raise ValueError(
            f"{field_name} must be a concrete WIDTHxHEIGHT value such as 1536x864; got {value!r}"
        )
    width, height = (int(part) for part in match.groups())
    if width <= 0 or height <= 0:
        raise ValueError(f"{field_name} dimensions must be positive; got {value!r}")
    if width * 9 != height * 16:
        raise ValueError(
            f"{field_name} must be exactly 16:9; got {width}x{height}"
        )
    return Size(width=width, height=height)


def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        raise FileNotFoundError(f"configuration file does not exist: {path}")
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key:
            values[key] = value
            os.environ.setdefault(key, value)
    return values


def _required(values: dict[str, str], key: str) -> str:
    value = values.get(key, "").strip()
    if not value:
        raise ValueError(f"missing required configuration: {key}")
    return value


def _positive_int(values: dict[str, str], key: str) -> int:
    try:
        value = int(_required(values, key))
    except ValueError as exc:
        raise ValueError(f"{key} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{key} must be positive")
    return value


def _nonnegative_float(values: dict[str, str], key: str) -> float:
    try:
        value = float(_required(values, key))
    except ValueError as exc:
        raise ValueError(f"{key} must be numeric") from exc
    if value < 0:
        raise ValueError(f"{key} must be non-negative")
    return value


def _value_or_default(values: dict[str, str], key: str, default: str) -> str:
    return values.get(key, "").strip() or default


def load_config(project_root: Path, *, env_path: Path | None = None) -> PipelineConfig:
    root = project_root.resolve()
    resolved_env = (env_path or root / ".env").resolve()
    values = load_env_file(resolved_env)
    validate_deck_font(DECK_FONT)
    return PipelineConfig(
        project_root=root,
        env_path=resolved_env,
        api_root=_required(values, "APIMART_API_ROOT").rstrip("/"),
        api_key_env="APIMART_API_KEY",
        chat_model=_required(values, "APIMART_CHAT_MODEL"),
        image_model=_required(values, "APIMART_IMAGE_MODEL"),
        image_size=parse_16_9_size(
            _required(values, "APIMART_IMAGE_SIZE"), field_name="APIMART_IMAGE_SIZE"
        ),
        image_resolution=_required(values, "APIMART_IMAGE_RESOLUTION"),
        image_quality=_required(values, "APIMART_IMAGE_QUALITY"),
        image_max_attempts=_positive_int(values, "APIMART_IMAGE_MAX_ATTEMPTS"),
        chat_max_tokens=_positive_int(values, "APIMART_CHAT_MAX_TOKENS"),
        vlm_max_tokens=_positive_int(values, "APIMART_VLM_MAX_TOKENS"),
        timeout_seconds=_nonnegative_float(values, "APIMART_TIMEOUT_SECONDS"),
        poll_initial_delay_seconds=_nonnegative_float(
            values, "APIMART_POLL_INITIAL_DELAY_SECONDS"
        ),
        poll_interval_seconds=_nonnegative_float(values, "APIMART_POLL_INTERVAL_SECONDS"),
        poll_timeout_seconds=_nonnegative_float(values, "APIMART_POLL_TIMEOUT_SECONDS"),
        diffusion_model_path=Path(
            _value_or_default(
                values, "DIFFUSION_MODEL_PATH", str(DEFAULT_DIFFUSION_MODEL_PATH)
            )
        ).expanduser(),
        diffusion_size=parse_16_9_size(
            _value_or_default(values, "DIFFUSION_IMAGE_SIZE", "1024x576"),
            field_name="DIFFUSION_IMAGE_SIZE",
        ),
        diffusion_steps=int(_value_or_default(values, "DIFFUSION_STEPS", "5")),
        diffusion_guidance_scale=float(
            _value_or_default(values, "DIFFUSION_GUIDANCE_SCALE", "0.0")
        ),
        diffusion_seed=int(_value_or_default(values, "DIFFUSION_SEED", "42")),
        deck_font=DECK_FONT,
    )
