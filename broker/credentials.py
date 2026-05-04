"""Host-only credential loading for the request broker."""

from __future__ import annotations

import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
INTEGRATION_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
SUPPORTED_CREDENTIAL_TYPES = {"bearer"}
SUPPORTED_METHODS = {"DELETE", "GET", "PATCH", "POST", "PUT"}
PROTECTED_DEFAULT_HEADERS = {
    "authorization",
    "cookie",
    "proxy-authorization",
    "set-cookie",
    "x-api-key",
}
REGEX_LIKE_PATH_TOKENS = ("[", "]", "(", ")", "{", "}", "+", "|", "^", "$", "\\")


class CredentialConfigError(ValueError):
    """Raised when host-only broker credentials are invalid."""


@dataclass(frozen=True, slots=True)
class HostCredential:
    """One host-only credential plus generic broker policy."""

    name: str
    credential_type: str
    token: str
    allowed_hosts: tuple[str, ...]
    allowed_methods: tuple[str, ...]
    allowed_paths: tuple[str, ...]
    default_headers: dict[str, str]

    def safe_metadata(self) -> dict[str, Any]:
        """Return model-visible metadata without secret material."""
        return {
            "name": self.name,
            "type": self.credential_type,
            "allowed_hosts": list(self.allowed_hosts),
            "allowed_methods": list(self.allowed_methods),
            "allowed_paths": list(self.allowed_paths),
        }


@dataclass(frozen=True, slots=True)
class HostCredentialRegistry:
    """Validated host-only credentials."""

    credentials: dict[str, HostCredential]

    def get(self, name: str) -> HostCredential | None:
        """Return one credential by name."""
        return self.credentials.get(name)

    def names(self) -> list[str]:
        """Return configured credential names in stable order."""
        return sorted(self.credentials)

    def safe_metadata(self) -> list[dict[str, Any]]:
        """Return safe metadata for prompt/MMDS use."""
        return [self.credentials[name].safe_metadata() for name in self.names()]


def default_credentials_path() -> Path:
    """Return the default host-only credentials file path."""
    return Path.home() / ".strangeclaw" / "secrets.yaml"


def load_host_credentials(
    path: Path | None = None,
    *,
    allow_missing: bool = True,
    allow_insecure_permissions: bool = False,
) -> HostCredentialRegistry:
    """Load and validate host-only request broker credentials."""
    credentials_path = path.expanduser() if path is not None else default_credentials_path()
    if not credentials_path.exists():
        if allow_missing:
            return HostCredentialRegistry(credentials={})
        raise CredentialConfigError(f"Host credentials file not found: {credentials_path}")

    _check_file_permissions(
        credentials_path,
        allow_insecure_permissions=allow_insecure_permissions,
    )
    loaded = _load_yaml_mapping(credentials_path)
    raw_credentials = loaded.get("credentials", {})
    if raw_credentials is None:
        raw_credentials = {}
    if not isinstance(raw_credentials, dict):
        raise CredentialConfigError("Host credentials field credentials must be a mapping.")

    credentials: dict[str, HostCredential] = {}
    for name, raw in raw_credentials.items():
        if not isinstance(name, str) or not INTEGRATION_NAME_RE.fullmatch(name):
            raise CredentialConfigError(
                "Host credentials keys must match ^[A-Za-z0-9_-]{1,64}$."
            )
        if not isinstance(raw, dict):
            raise CredentialConfigError(f"Host credentials.{name} must be a mapping.")
        credentials[name] = _parse_credential(name, raw)

    return HostCredentialRegistry(credentials=credentials)


def _check_file_permissions(
    path: Path,
    *,
    allow_insecure_permissions: bool,
) -> None:
    if os.name != "posix" or allow_insecure_permissions:
        return
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise CredentialConfigError(
            f"Host credentials file {path} must not be group/world-readable. "
            "Set permissions to 0600 or stricter."
        )


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise CredentialConfigError(f"Invalid YAML in host credentials file {path}: {exc}") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise CredentialConfigError(
            f"Host credentials file {path} must contain a top-level mapping."
        )
    return cast(dict[str, Any], loaded)


def _parse_credential(name: str, raw: dict[str, Any]) -> HostCredential:
    credential_type = raw.get("type", "bearer")
    if not isinstance(credential_type, str):
        raise CredentialConfigError(f"Host credentials.{name}.type must be a string.")
    credential_type = credential_type.strip().lower()
    if credential_type not in SUPPORTED_CREDENTIAL_TYPES:
        allowed = ", ".join(sorted(SUPPORTED_CREDENTIAL_TYPES))
        raise CredentialConfigError(
            f"Host credentials.{name}.type must be one of: {allowed}."
        )

    token_raw = raw.get("token")
    if not isinstance(token_raw, str):
        raise CredentialConfigError(f"Host credentials.{name}.token must be a string.")
    token = _resolve_env_vars(token_raw, field=f"credentials.{name}.token").strip()
    if not token:
        raise CredentialConfigError(f"Host credentials.{name}.token must be non-empty.")

    return HostCredential(
        name=name,
        credential_type=credential_type,
        token=token,
        allowed_hosts=_parse_allowed_hosts(name, raw.get("allowed_hosts")),
        allowed_methods=_parse_allowed_methods(name, raw.get("allowed_methods")),
        allowed_paths=_parse_allowed_paths(name, raw.get("allowed_paths")),
        default_headers=_parse_default_headers(name, raw.get("default_headers", {})),
    )


def _resolve_env_vars(value: str, *, field: str) -> str:
    def replace(match: re.Match[str]) -> str:
        var_name = match.group(1)
        env_value = os.getenv(var_name)
        if env_value is None:
            raise CredentialConfigError(
                f"Missing environment variable '{var_name}' required by host "
                f"credentials field '{field}'."
            )
        return env_value

    return ENV_PATTERN.sub(replace, value)


def _parse_allowed_hosts(name: str, raw: Any) -> tuple[str, ...]:
    values = _require_non_empty_string_list(raw, field=f"credentials.{name}.allowed_hosts")
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_host in values:
        host = raw_host.strip().lower()
        if not _is_valid_hostname(host):
            raise CredentialConfigError(
                f"Host credentials.{name}.allowed_hosts contains invalid host "
                f"'{raw_host}'. Hosts must not include scheme, port, path, or credentials."
            )
        if host not in seen:
            seen.add(host)
            normalized.append(host)
    return tuple(normalized)


def _parse_allowed_methods(name: str, raw: Any) -> tuple[str, ...]:
    values = _require_non_empty_string_list(raw, field=f"credentials.{name}.allowed_methods")
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_method in values:
        method = raw_method.strip().upper()
        if method not in SUPPORTED_METHODS:
            allowed = ", ".join(sorted(SUPPORTED_METHODS))
            raise CredentialConfigError(
                f"Host credentials.{name}.allowed_methods contains unsupported method "
                f"'{raw_method}'. Supported methods: {allowed}."
            )
        if method not in seen:
            seen.add(method)
            normalized.append(method)
    return tuple(normalized)


def _parse_allowed_paths(name: str, raw: Any) -> tuple[str, ...]:
    values = _require_non_empty_string_list(raw, field=f"credentials.{name}.allowed_paths")
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_path in values:
        path = raw_path.strip()
        if not _is_valid_path_pattern(path):
            raise CredentialConfigError(
                f"Host credentials.{name}.allowed_paths contains invalid pattern "
                f"'{raw_path}'. Use absolute paths with optional trailing '*' only."
            )
        if path not in seen:
            seen.add(path)
            normalized.append(path)
    return tuple(normalized)


def _parse_default_headers(name: str, raw: Any) -> dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise CredentialConfigError(
            f"Host credentials.{name}.default_headers must be a mapping."
        )

    headers: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not _is_valid_header_name(key):
            raise CredentialConfigError(
                f"Host credentials.{name}.default_headers keys must be valid "
                "HTTP header names."
            )
        if key.strip().lower() in PROTECTED_DEFAULT_HEADERS:
            raise CredentialConfigError(
                f"Host credentials.{name}.default_headers must not include "
                f"protected header '{key}'."
            )
        if not isinstance(value, str):
            raise CredentialConfigError(
                f"Host credentials.{name}.default_headers.{key} must be a string."
            )
        headers[key.strip()] = value
    return headers


def _require_non_empty_string_list(raw: Any, *, field: str) -> list[str]:
    if not isinstance(raw, list) or not raw:
        raise CredentialConfigError(f"Host credentials {field} must be a non-empty list.")
    values: list[str] = []
    for index, item in enumerate(raw):
        if not isinstance(item, str) or not item.strip():
            raise CredentialConfigError(
                f"Host credentials {field} must contain non-empty strings "
                f"(invalid at index {index})."
            )
        values.append(item)
    return values


def _is_valid_hostname(value: str) -> bool:
    if not value or len(value) > 253:
        return False
    if any(char in value for char in "/:@[]"):
        return False
    labels = value.split(".")
    for label in labels:
        if not label or len(label) > 63:
            return False
        if label.startswith("-") or label.endswith("-"):
            return False
        if not all(char.isalnum() or char == "-" for char in label):
            return False
    return True


def _is_valid_path_pattern(value: str) -> bool:
    if not value.startswith("/"):
        return False
    if "?" in value or "#" in value:
        return False
    if any(token in value for token in REGEX_LIKE_PATH_TOKENS):
        return False
    star_count = value.count("*")
    if star_count == 0:
        return True
    return star_count == 1 and value.endswith("*")


def _is_valid_header_name(value: str) -> bool:
    stripped = value.strip()
    return bool(stripped) and all(char.isalnum() or char in "!#$%&'*+-.^_`|~" for char in stripped)
