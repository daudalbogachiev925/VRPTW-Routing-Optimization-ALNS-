"""Configuration loader."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class OperatorSpec(BaseModel):
    name: str
    weight: float = 1.0
    params: dict[str, Any] = Field(default_factory=dict)


class OperatorsConfig(BaseModel):
    destroy: list[OperatorSpec] = Field(default_factory=list)
    repair: list[OperatorSpec] = Field(default_factory=list)


class ALNSConfig(BaseModel):
    iterations: int = 30000
    segment_length: int = 100
    reaction_factor: float = 0.1
    score_new_best: float = 33.0
    score_better: float = 9.0
    score_accepted: float = 13.0
    score_rejected: float = 0.0
    sa_initial_temp: float = 100.0
    sa_cooling: float = 0.99975
    seed: int = 42


class BenchmarkConfig(BaseModel):
    instances_dir: str = "./data/solomon"
    instances: list[str] = Field(default_factory=lambda: ["c101", "c201", "r101", "rc101"])
    default_iterations: int = 30000
    output_dir: str = "./artifacts"


class AppConfig(BaseModel):
    alns: ALNSConfig = Field(default_factory=ALNSConfig)
    operators: OperatorsConfig = Field(default_factory=OperatorsConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"
    artifacts_dir: str = "./artifacts"
    config_path: str = "./configs/config.yaml"

    def load_app_config(self) -> AppConfig:
        path = Path(self.config_path)
        if not path.exists():
            return AppConfig()
        with path.open("r", encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
        return AppConfig(**raw)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@lru_cache(maxsize=1)
def get_app_config() -> AppConfig:
    return get_settings().load_app_config()


def ensure_dirs() -> None:
    Path(get_settings().artifacts_dir).mkdir(parents=True, exist_ok=True)
