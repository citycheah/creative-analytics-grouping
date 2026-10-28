from dataclasses import dataclass, fields
import json
from typing import Any

import pandas as pd

from .exceptions import ConfigError
from .models import AdAnalysisResults


@dataclass
class AdAnalyserConfig:
    df: pd.DataFrame | None = None
    input_query: str | None = None
    gcp_project_id: str | None = None
    gcp_region: str | None = None
    gcs_bucket: str | None = None
    gcs_prefix: str | None = None
    dest_table: str | None = None
    output_path: str | None = None
    include_logs: bool = False
    model: str = "gemini-2.5-flash"
    system_instruction: str | None = None
    response_schema: Any = AdAnalysisResults
    params: dict[str, Any] | None = None
    labels: dict[str, str] | None = None
    chunk_size: int = 1000
    max_workers: int = 25

    @classmethod
    def from_json(cls, file_path: str, **kwargs) -> "AdAnalyserConfig":
        with open(file_path) as f:
            data = json.load(f)
        data.update(kwargs)
        valid_keys = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)

    @classmethod
    def from_object(cls, obj: Any, **kwargs) -> "AdAnalyserConfig":
        data = {}
        for k in dir(obj):
          if not k.startswith('_'):
              val = getattr(obj, k)
              if not callable(val):
                  data[k] = val
        valid_keys = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        filtered_data.update(kwargs)
        return cls(**filtered_data)

    def __post_init__(self):
        if self.df is not None and self.input_query is not None:
            raise ConfigError("Provide either 'df' or 'input_query', not both.")
        if self.params is None:
            self.params = {}
