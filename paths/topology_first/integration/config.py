from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


_SIZE_PATTERN = re.compile(r"^\s*(\d+)\s*[xX\u00d7]\s*(\d+)\s*$")
DECK_FONT = "Noto Sans"


@lru_cache(maxsize=None)
def validate_deck_font(family: str = DECK_FONT) -> dict[str, str]:
    """Resolve actual font files; a fontconfig substitute is not this family."""
    files = {}
    for style in ("Regular", "Bold"):
        result = subprocess.run(
            ["fc-match", "-f", "%{family}\n%{style}\n%{file}", f"{family}:style={style}"],
            capture_output=True, text=True, check=False,
        )
        parts = result.stdout.strip().splitlines()
        if result.returncode or len(parts) != 3 or family not in {s.strip() for s in parts[0].split(",")}:
            raise RuntimeError(f"required deck font is not installed: {family} {style}")
        if style == "Bold" and "bold" not in parts[1].casefold():
            raise RuntimeError(f"required deck font weight is not installed: {family} {style}")
        if not Path(parts[2]).is_file():
            raise RuntimeError(f"deck font file is unavailable: {family} {style}")
        files[style.casefold()] = parts[2]
    return files


@lru_cache(maxsize=None)
def resolve_deck_font(requested: str) -> str:
    try:
        validate_deck_font(requested)
        return requested
    except RuntimeError:
        validate_deck_font(DECK_FONT)
        return DECK_FONT


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
    title_font: str
    body_font: str
    cjk_font: str


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


def load_config(project_root: Path, *, env_path: Path | None = None) -> PipelineConfig:
    root = project_root.resolve()
    resolved_env = (env_path or root / ".env").resolve()
    values = load_env_file(resolved_env)
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
        diffusion_model_path=Path(_required(values, "DIFFUSION_MODEL_PATH")).expanduser(),
        diffusion_size=parse_16_9_size(
            _required(values, "DIFFUSION_IMAGE_SIZE"), field_name="DIFFUSION_IMAGE_SIZE"
        ),
        diffusion_steps=_positive_int(values, "DIFFUSION_STEPS"),
        diffusion_guidance_scale=_nonnegative_float(values, "DIFFUSION_GUIDANCE_SCALE"),
        diffusion_seed=int(_required(values, "DIFFUSION_SEED")),
        title_font=resolve_deck_font(values.get("PPT_TITLE_FONT") or DECK_FONT),
        body_font=resolve_deck_font(values.get("PPT_BODY_FONT") or DECK_FONT),
        cjk_font=_required(values, "PPT_CJK_FONT"),
    )
