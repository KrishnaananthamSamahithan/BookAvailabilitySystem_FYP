from dataclasses import dataclass
from typing import Dict, List, Optional

from src.utils import normalize_name


@dataclass
class SchemaResolution:
    required: Dict[str, str]
    optional: Dict[str, Optional[str]]
    missing_required: List[str]
    matched_aliases: Dict[str, Optional[str]]

    @property
    def is_valid(self) -> bool:
        return not self.missing_required

    def all_columns(self) -> Dict[str, Optional[str]]:
        combined = dict(self.optional)
        combined.update(self.required)
        return combined


def resolve_schema(columns: List[str], aliases: Dict[str, List[str]], required_fields: List[str], optional_fields: List[str]) -> SchemaResolution:
    normalized_columns = {normalize_name(column): column for column in columns}

    required: Dict[str, str] = {}
    optional: Dict[str, Optional[str]] = {}
    missing_required: List[str] = []
    matched_aliases: Dict[str, Optional[str]] = {}

    for logical_name in required_fields:
        match, matched_alias = _resolve_single(logical_name, normalized_columns, aliases)
        if match is None:
            missing_required.append(logical_name)
        else:
            required[logical_name] = match
        matched_aliases[logical_name] = matched_alias

    for logical_name in optional_fields:
        match, matched_alias = _resolve_single(logical_name, normalized_columns, aliases)
        optional[logical_name] = match
        matched_aliases[logical_name] = matched_alias

    return SchemaResolution(required=required, optional=optional, missing_required=missing_required, matched_aliases=matched_aliases)


def _resolve_single(logical_name: str, normalized_columns: Dict[str, str], aliases: Dict[str, List[str]]) -> tuple[Optional[str], Optional[str]]:
    candidates = aliases.get(logical_name, []) + [logical_name]
    for candidate in candidates:
        normalized = normalize_name(candidate)
        if normalized in normalized_columns:
            return normalized_columns[normalized], candidate
    return None, None
