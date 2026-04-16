import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def safe_to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", format="mixed")


def safe_to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def normalize_text(series: pd.Series, default: str = "Unknown") -> pd.Series:
    cleaned = series.astype("string").fillna(default).str.strip()
    cleaned = cleaned.replace({"": default, "<NA>": default, "nan": default, "None": default})
    return cleaned.fillna(default)


def extract_domain(value: Any, default: str = "Unknown") -> str:
    if pd.isna(value):
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        parsed = urlparse(text)
        domain = parsed.netloc.lower().replace("www.", "")
        return domain or default
    except Exception:
        return default


def parse_query_params(url: Any) -> Dict[str, str]:
    if pd.isna(url):
        return {}
    try:
        parsed = urlparse(str(url))
        parsed_qs = parse_qs(parsed.query)
        return {k.lower(): v[0] for k, v in parsed_qs.items() if v}
    except Exception:
        return {}


def first_present(params: Dict[str, str], keys: Iterable[str], default: Optional[str] = None) -> Optional[str]:
    for key in keys:
        if key in params and params[key] not in (None, ""):
            return params[key]
    return default


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)


def save_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
