"""HTTP client for the weekly forecast endpoint (Databricks Model Serving).

- Sends series to the endpoint with timeouts and retries with growing waits (3, 6, 12 s).
- Classifies errors: invalid input, authentication, missing endpoint, unavailable service.
- Validates the response before using it: fields, list lengths, finite non-negative numbers,
  ordered quantiles and ISO dates.
It contains no model: every forecast comes from a request to the published endpoint.
"""
import math
import time
from datetime import date

import requests

REQUIRED = ("series_id", "status", "scale", "target_week", "week_start", "mean", "p10", "p50", "p90",
            "model_version")
RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}
DEFAULT_TIMEOUT = (20, 90)
DEFAULT_ATTEMPTS = 4
MAX_BACKOFF_SECONDS = 20
MAX_SERIES_PER_REQUEST = 200   # service limit


class ApiError(Exception):
    user_message = "The model request could not be completed."

    def __init__(self, detail="", status=None):
        super().__init__(detail)
        self.detail = detail
        self.status = status


class InvalidInput(ApiError):
    user_message = "The service rejected the input data. Fix the values and try again."


class AuthError(ApiError):
    user_message = "Authentication with the model failed: the token is invalid or has expired."


class NotFound(ApiError):
    user_message = "The model endpoint was not found. Check the endpoint name."


class Unavailable(ApiError):
    user_message = "The model did not respond in time. It may be starting up: wait a minute and try again."


class InvalidResponse(ApiError):
    user_message = "The service returned a format this app cannot use."


class ConfigError(ApiError):
    user_message = "The model connection is not configured."


def _error_message(response):
    try:
        body = response.json()
        if isinstance(body, dict):
            return str(body.get("message") or body.get("error") or response.text[:300])
    except ValueError:
        pass
    return response.text[:300].strip() or "no detail"


def classify_error(response):
    status = response.status_code
    message = _error_message(response)
    if status == 400:
        return InvalidInput(message, status)
    if status in (401, 403):
        return AuthError(message, status)
    if status == 404:
        return NotFound(message, status)
    if status in RETRY_STATUS:
        return Unavailable(f"The service returned HTTP {status}: {message}", status)
    return ApiError(f"HTTP {status}: {message}", status)


def _finite_list(values, n, name, sid, status):
    if not isinstance(values, list) or len(values) != n:
        raise InvalidResponse(f"{sid}: '{name}' must be a list of {n} values.", status)
    out = []
    for v in values:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
            raise InvalidResponse(f"{sid}: '{name}' has an invalid value ({str(v)[:30]}).", status)
        out.append(float(v))
    return out


def parse_forecasts(body, expected_ids, status=200):
    """Validates the body of a 200 response and returns the forecasts (one per requested series)."""
    preds = body.get("predictions") if isinstance(body, dict) else None
    if not isinstance(preds, list) or len(preds) != len(expected_ids):
        raise InvalidResponse(f"Expected {len(expected_ids)} forecasts but received: {str(preds)[:120]}", status)
    clean = []
    for sid_expected, item in zip(expected_ids, preds):
        if not isinstance(item, dict) or any(k not in item for k in REQUIRED):
            missing = [k for k in REQUIRED if not isinstance(item, dict) or k not in item]
            raise InvalidResponse(f"{sid_expected}: missing fields {missing}.", status)
        sid = item["series_id"]
        if sid != sid_expected:
            raise InvalidResponse(f"Response order mismatch ({sid} instead of {sid_expected}).", status)
        n = len(item["mean"]) if isinstance(item["mean"], list) else -1
        if n < 1:
            raise InvalidResponse(f"{sid}: 'mean' is empty or malformed.", status)
        rec = {"series_id": sid, "status": str(item["status"]), "model_version": str(item["model_version"]),
               "scale": _scalar(item["scale"], sid, status)}
        for name in ("mean", "p10", "p50", "p90"):
            rec[name] = _finite_list(item[name], n, name, sid, status)
        if not isinstance(item["target_week"], list) or len(item["target_week"]) != n:
            raise InvalidResponse(f"{sid}: 'target_week' must have {n} values.", status)
        rec["target_week"] = [int(w) for w in item["target_week"]]
        try:
            rec["week_start"] = [date.fromisoformat(str(d)) for d in item["week_start"]]
        except (ValueError, TypeError):
            raise InvalidResponse(f"{sid}: 'week_start' has no valid ISO dates.", status)
        if len(rec["week_start"]) != n:
            raise InvalidResponse(f"{sid}: 'week_start' must have {n} values.", status)
        eps = 1e-6
        if any(a > b + eps or b > c + eps for a, b, c in zip(rec["p10"], rec["p50"], rec["p90"])):
            raise InvalidResponse(f"{sid}: quantiles are not ordered (P10 ≤ P50 ≤ P90).", status)
        clean.append(rec)
    return clean


def _scalar(v, sid, status):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
        raise InvalidResponse(f"{sid}: 'scale' is not valid.", status)
    return float(v)


def query_endpoint(rows, host, token, endpoint, max_attempts=DEFAULT_ATTEMPTS, timeout=DEFAULT_TIMEOUT,
                   sleep=time.sleep, session=None):
    """Queries the endpoint and returns (forecasts, info).

    rows: list of dicts with series_id, level, last_week, horizon, history.
    Retries only when it makes sense: timeout, connection error, 408/425/429 and 5xx.
    Does not retry 400, 401, 403 or 404.
    """
    if not str(host).startswith("https://") and not str(host).startswith("http://127.0.0.1"):
        raise ConfigError("DATABRICKS_HOST must start with https://")
    if not token or not endpoint:
        raise ConfigError("DATABRICKS_TOKEN or ENDPOINT_NAME is missing.")
    if not rows:
        raise InvalidInput("There are no series to query.")
    if len(rows) > MAX_SERIES_PER_REQUEST:
        raise InvalidInput(f"At most {MAX_SERIES_PER_REQUEST} series per request (sent {len(rows)}).")

    http = session or requests
    url = f"{str(host).rstrip('/')}/serving-endpoints/{endpoint}/invocations"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    ids = [r["series_id"] for r in rows]
    started = time.perf_counter()
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = http.post(url, headers=headers, json={"dataframe_records": rows}, timeout=timeout)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            last_error = Unavailable(f"{type(exc).__name__}: the service did not respond.")
        except requests.exceptions.RequestException as exc:
            raise ApiError(f"{type(exc).__name__}: {str(exc)[:200]}")
        else:
            if response.status_code == 200:
                try:
                    body = response.json()
                except ValueError:
                    raise InvalidResponse("The response is not JSON.", 200)
                forecasts = parse_forecasts(body, ids, 200)
                return forecasts, {"attempts": attempt, "seconds": time.perf_counter() - started,
                                   "series": len(rows)}
            error = classify_error(response)
            if not isinstance(error, Unavailable):
                raise error
            last_error = error
        if attempt < max_attempts:
            sleep(min(3 * 2 ** (attempt - 1), MAX_BACKOFF_SECONDS))
    last_error.detail = f"{last_error.detail} (after {max_attempts} attempts)"
    raise last_error


def query_many(rows, host, token, endpoint, **kwargs):
    """Queries in blocks of up to 200 series and joins the responses."""
    out, total, attempts = [], 0.0, 0
    for i in range(0, len(rows), MAX_SERIES_PER_REQUEST):
        part, info = query_endpoint(rows[i:i + MAX_SERIES_PER_REQUEST], host, token, endpoint, **kwargs)
        out += part
        total += info["seconds"]
        attempts = max(attempts, info["attempts"])
    return out, {"attempts": attempts, "seconds": total, "series": len(rows)}
