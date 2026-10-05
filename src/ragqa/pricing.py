"""USD prices per 1M tokens, used for cost accounting in traces and evaluation reports.

Prices change; verify against the provider's pricing page and override with
``RAGQA_PRICES='{"model": [input_per_1m, output_per_1m]}'`` when needed.
"""

from __future__ import annotations

import json
import logging
import os

log = logging.getLogger(__name__)

# (input $/1M tokens, output $/1M tokens) - checked October 2026.
_DEFAULT_PRICES: dict[str, tuple[float, float]] = {
    "gpt-6-luna": (0.10, 0.50),
    "gpt-4o-mini": (0.15, 0.60),
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
}


def _load_prices() -> dict[str, tuple[float, float]]:
    prices = dict(_DEFAULT_PRICES)
    raw = os.environ.get("RAGQA_PRICES")
    if raw:
        try:
            for model, pair in json.loads(raw).items():
                prices[model] = (float(pair[0]), float(pair[1]))
        except (ValueError, TypeError, IndexError, KeyError):
            log.warning("Ignoring malformed RAGQA_PRICES value")
    return prices


PRICES = _load_prices()
_warned: set[str] = set()


def cost_usd(model: str, input_tokens: int, output_tokens: int = 0) -> float:
    price = PRICES.get(model)
    if price is None:
        if model not in _warned:
            log.info("No price configured for model %s; cost reported as 0", model)
            _warned.add(model)
        return 0.0
    return (input_tokens * price[0] + output_tokens * price[1]) / 1_000_000
