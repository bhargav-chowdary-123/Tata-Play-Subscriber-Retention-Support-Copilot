"""Reliable client for the offline mock chat-completion API."""

import os
import time

import mock_llm
import requests

BASE_DELAY = 0.05
MODEL = "openai/gpt-4o-mini"

_rate_limit_recoveries = 0


def reset_stats() -> None:
    """Reset the counter of recovered rate-limited responses."""
    global _rate_limit_recoveries
    _rate_limit_recoveries = 0


def stats() -> int:
    """Return the number of rate-limited responses recovered by retries."""
    return _rate_limit_recoveries


def call_llm(messages, temperature: float = 0.0, max_tokens: int = 300) -> str:
    """Call the configured chat-completion endpoint and return its text.

    Uses the offline mock by default. When LLM_MODE=live, uses requests.post
    against LLM_URL with an API key read from LLM_API_KEY. Retries only HTTP
    429 responses, at most three times.
    """
    global _rate_limit_recoveries

    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if os.environ.get("LLM_MODE") == "live":
        url = os.environ["LLM_URL"]
        post = requests.post
        api_key = os.environ["LLM_API_KEY"]
    else:
        url = mock_llm.URL
        post = mock_llm.post
        api_key = "mock-key"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for attempt in range(4):
        response = post(url, headers=headers, json=payload, timeout=30)

        if response.status_code == 200:
            body = response.json()
            choice = body["choices"][0]
            if choice["finish_reason"] == "length":
                print("Warning: response truncated because max_tokens was reached.")
            return choice["message"]["content"]

        if response.status_code == 429 and attempt < 3:
            _rate_limit_recoveries += 1
            time.sleep(BASE_DELAY * (2 ** attempt))
            continue

        raise RuntimeError(
            f"LLM request failed with status {response.status_code}: {response.text}"
        )

    raise RuntimeError("LLM request failed after the maximum number of retries.")
