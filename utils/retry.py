import json
import logging
import os
from typing import List, Dict, Callable, Any
from joblib import Memory
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)
memory = Memory(".cache", verbose=0)  # cache dir

# Cost tracking
total_tokens_used = 0
total_api_calls = 0
total_cost = 0.0

# Model pricing in $/Million tokens
MODEL_PRICING = {
    "gpt-5": {"input": 1.250, "output": 10.000},  # $1.25/$10.00 per 1M tokens
    "gpt-5-mini": {"input": 0.250, "output": 2.000},  # $0.25/$2.00 per 1M tokens
    "gpt-5-nano": {"input": 0.050, "output": 0.400},  # $0.05/$0.40 per 1M tokens
}


def get_model_pricing(model_id: str) -> tuple:
    """Get input and output pricing for a model."""
    if model_id not in MODEL_PRICING:
        raise ValueError(
            f"Unsupported model: {model_id}. Supported models: {list(MODEL_PRICING.keys())}"
        )
    pricing = MODEL_PRICING[model_id]
    return pricing["input"], pricing["output"]


def make_hashable(messages: List[Dict[str, str]]) -> str:
    # Ensure stable hashing by sorting keys
    return json.dumps(messages, sort_keys=True)


def _make_api_call(messages_json: str, model_id: str) -> str:
    """Raw OpenAI call without caching."""
    global total_tokens_used, total_api_calls, total_cost

    messages = json.loads(messages_json)
    response = openai.chat.completions.create(
        model=model_id,
        messages=messages,
        response_format={"type": "json_object"},
        reasoning_effort="minimal",
    )

    # Track costs
    total_api_calls += 1
    usage = response.usage
    if usage:
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens
        total_tokens_used += input_tokens + output_tokens

        # Calculate cost using model-specific pricing (per million tokens)
        input_price, output_price = get_model_pricing(model_id)
        input_cost = (input_tokens / 1_000_000) * input_price
        output_cost = (output_tokens / 1_000_000) * output_price
        call_cost = input_cost + output_cost
        total_cost += call_cost

        logger.debug(
            f"API call #{total_api_calls} ({model_id}): {input_tokens} input + {output_tokens} output tokens = ${call_cost:.4f}"
        )

    return response.choices[0].message.content


@memory.cache
def _cached_api_call(messages_json: str, model_id: str) -> str:
    """Raw OpenAI call, cached by joblib."""
    return _make_api_call(messages_json, model_id)


def _uncached_api_call(messages_json: str, model_id: str) -> str:
    """Raw OpenAI call without caching."""
    return _make_api_call(messages_json, model_id)


def retry_with_fallback(
    messages: List[Dict[str, str]],
    validation_func: Callable[[str], bool],
    max_retries: int = 5,
    fallback_value: Any = None,
    model_id: str = "gpt-5-mini",
) -> Any:
    messages_json = make_hashable(messages)

    # Check environment variable for cache setting (defaults to True)
    use_cache = os.getenv("USE_CACHE", "True").lower() in ("true", "1", "yes")

    for attempt in range(max_retries):
        try:
            if use_cache:
                content = _cached_api_call(messages_json, model_id)
            else:
                content = _uncached_api_call(messages_json, model_id)
        except Exception as e:
            logger.error(f"API call failed on attempt {attempt+1}: {e}")
            continue

        if validation_func(content):
            return content
        else:
            logger.warning(f"Invalid response on attempt {attempt+1}, retrying...")

    logger.error(f"Failed after {max_retries} attempts")
    return fallback_value


def get_cost_summary() -> Dict[str, Any]:
    """Get current cost summary."""
    return {
        "total_api_calls": total_api_calls,
        "total_tokens_used": total_tokens_used,
        "total_cost": total_cost,
        "avg_cost_per_call": total_cost / total_api_calls if total_api_calls > 0 else 0,
        "avg_tokens_per_call": (
            total_tokens_used / total_api_calls if total_api_calls > 0 else 0
        ),
    }


def print_cost_report():
    """Print a detailed cost report."""
    summary = get_cost_summary()

    print("\n" + "=" * 60)
    print("API COST REPORT")
    print("=" * 60)
    print(f"Total API Calls: {summary['total_api_calls']:,}")
    print(f"Total Tokens Used: {summary['total_tokens_used']:,}")
    print(f"Total Cost: ${summary['total_cost']:.4f}")
    print(f"Average Cost per Call: ${summary['avg_cost_per_call']:.4f}")
    print(f"Average Tokens per Call: {summary['avg_tokens_per_call']:.1f}")
    if summary["total_cost"] > 0:
        print(
            f"Cost per 1M Tokens: ${(summary['total_cost'] / summary['total_tokens_used'] * 1_000_000):.2f}"
        )
    print("=" * 60)

    # Log the summary as well
    logger.info(
        f"API Cost Summary: {summary['total_api_calls']} calls, ${summary['total_cost']:.4f} total cost"
    )


def reset_cost_tracking():
    """Reset cost tracking counters."""
    global total_tokens_used, total_api_calls, total_cost
    total_tokens_used = 0
    total_api_calls = 0
    total_cost = 0.0
