import json
import logging
import os
from typing import List, Dict, Callable, Any, Optional, Tuple
from joblib import Memory
import openai
from dotenv import load_dotenv
from vllm import LLM, SamplingParams
from copy import deepcopy
import google.generativeai as genai

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
    # Gemini 2.5 Pro - Standard pricing
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},  # $1.25/$10.00 per 1M tokens (prompts <= 200k)
    # Gemini 2.5 Flash - Standard pricing  
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},  # $0.30/$2.50 per 1M tokens
    "gemini-2.5-flash-lite": {"input": 0.075, "output": 0.30},  # $0.075/$0.30 per 1M tokens
    # Gemini 2.0 Flash - Standard pricing
    "gemini-2.0-flash": {"input": 0.075, "output": 0.30},  # $0.075/$0.30 per 1M tokens
    "gemini-2.0-flash-lite": {"input": 0.0375, "output": 0.15},  # $0.0375/$0.15 per 1M tokens
    "gemini-2.0-flash-exp": {"input": 0.075, "output": 0.30},  # $0.075/$0.30 per 1M tokens
    # Gemini 1.5 Flash - Standard pricing
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},  # $0.075/$0.30 per 1M tokens
    "gemini-1.5-flash-8b": {"input": 0.0375, "output": 0.15},  # $0.0375/$0.15 per 1M tokens
    # Gemini 1.5 Pro - Standard pricing
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},  # $1.25/$5.00 per 1M tokens
    "gemini-1.5-pro-128k": {"input": 1.25, "output": 5.00},  # $1.25/$5.00 per 1M tokens
}


def is_gpt_model(model_id: str) -> bool:
    """Check if the model is a GPT model (OpenAI)."""
    return model_id.startswith("gpt-")


def is_gemini_model(model_id: str) -> bool:
    """Check if the model is a Gemini model (Google)."""
    return model_id.startswith("gemini-")


def get_model_pricing(model_id: str) -> tuple:
    """Get input and output pricing for a model."""
    if model_id not in MODEL_PRICING:
        # For non-GPT models, return 0 cost (or could raise an error)
        logger.warning(
            f"No pricing info for model: {model_id}, assuming free/local model"
        )
        return 0.0, 0.0
    pricing = MODEL_PRICING[model_id]
    return pricing["input"], pricing["output"]


def make_hashable(messages: List[Dict[str, str]]) -> str:
    # Ensure stable hashing by sorting keys
    return json.dumps(messages, sort_keys=True)


def _vllm_api_call(messages: List[Dict[str, str]], model_id: str) -> str:
    """Make an API call using vLLM for HuggingFace models."""
    # Prevent debugger from attaching to vLLM subprocesses (fixes Python 3.12 'imp' module error)
    # This is necessary because vLLM spawns subprocesses for model inspection,
    # and the VS Code debugger (debugpy) has compatibility issues with Python 3.12
    os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
    if "PYTHONBREAKPOINT" in os.environ:
        del os.environ["PYTHONBREAKPOINT"]

    llm = LLM(model=model_id, trust_remote_code=True, disable_log_stats=True)

    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=2048,
    )

    # Use chat method with OpenAI-compatible format
    outputs = llm.chat(messages=[messages], sampling_params=sampling_params)
    response_text = outputs[0].outputs[0].text.strip()

    logger.debug(f"vLLM call completed ({model_id})")

    return response_text


def _track_api_cost(input_tokens: int, output_tokens: int, model_id: str, estimated: bool = False) -> None:
    """Track API call costs and update global counters."""
    global total_tokens_used, total_api_calls, total_cost
    
    total_api_calls += 1
    total_tokens_used += input_tokens + output_tokens
    
    # Calculate cost using model-specific pricing (per million tokens)
    input_price, output_price = get_model_pricing(model_id)
    input_cost = (input_tokens / 1_000_000) * input_price
    output_cost = (output_tokens / 1_000_000) * output_price
    call_cost = input_cost + output_cost
    total_cost += call_cost
    
    token_prefix = "~" if estimated else ""
    logger.debug(
        f"API call #{total_api_calls} ({model_id}): {token_prefix}{input_tokens} input + {token_prefix}{output_tokens} output tokens = ${call_cost:.4f}"
    )


def _gemini_api_call(messages: List[Dict[str, str]], model_id: str) -> Tuple[str, int, int]:
    """Make an API call using Google Gemini API with JSON response format.
    
    Returns:
        Tuple of (response_text, input_tokens, output_tokens)
    """
    # Configure the API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    genai.configure(api_key=api_key)
    
    # Convert OpenAI format messages to Gemini format
    # Gemini expects a single string or a list of content parts
    if len(messages) == 1 and messages[0]["role"] == "user":
        # Simple user message
        prompt = messages[0]["content"]
    else:
        # Convert conversation format
        prompt_parts = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
            elif role == "system":
                prompt_parts.append(f"System: {content}")
        prompt = "\n".join(prompt_parts)
    
    # Generate content using Gemini with JSON response format
    model = genai.GenerativeModel(model_id)
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.0,
            max_output_tokens=2048,
            response_mime_type="application/json",
        )
    )
    
    logger.debug(f"Gemini call completed ({model_id})")
    
    # Extract token counts from response metadata
    input_tokens = response.usage_metadata.prompt_token_count if response.usage_metadata else 0
    output_tokens = response.usage_metadata.candidates_token_count if response.usage_metadata else 0
    
    return response.text, input_tokens, output_tokens


def update_cost_from_summary(summary: Dict[str, Any]) -> None:
    """Update global cost tracking from a summary dictionary."""
    global total_tokens_used, total_api_calls, total_cost
    
    total_tokens_used += summary.get("total_tokens_used", 0)
    total_api_calls += summary.get("total_api_calls", 0)
    total_cost += summary.get("total_cost", 0.0)



def _make_api_call(messages_json: str, model_id: str) -> str:
    """Route to OpenAI, Gemini, or vLLM based on model type."""
    global total_tokens_used, total_api_calls, total_cost

    messages = json.loads(messages_json)

    # Check if this is a GPT model
    if is_gpt_model(model_id):
        # Use OpenAI API
        response = openai.chat.completions.create(
            model=model_id,
            messages=messages,
            response_format={"type": "json_object"},
            reasoning_effort="low",
        )

        # Track costs
        usage = response.usage
        if usage:
            input_tokens = usage.prompt_tokens
            output_tokens = usage.completion_tokens
            _track_api_cost(input_tokens, output_tokens, model_id, estimated=False)

        return response.choices[0].message.content
    elif is_gemini_model(model_id):
        # Use Gemini API
        response_text, input_tokens, output_tokens = _gemini_api_call(messages, model_id)
        
        # Track costs using actual token counts from Gemini response
        _track_api_cost(input_tokens, output_tokens, model_id, estimated=False)
        
        return response_text
    else:
        # Use vLLM for HuggingFace models
        return _vllm_api_call(messages, model_id)


@memory.cache
def _cached_api_call(messages_json: str, model_id: str) -> str:
    """Raw OpenAI call, cached by joblib."""
    logger.debug(f"Cached API call ({model_id})")
    return _make_api_call(messages_json, model_id)


def _uncached_api_call(messages_json: str, model_id: str) -> str:
    """Raw OpenAI call without caching."""
    return _make_api_call(messages_json, model_id)


def retry_with_fallback(
    messages: List[Dict[str, str]],
    validation_func: Callable[[str], bool],
    model_id: str,
    max_retries: int = 5,
    fallback_value: Any = None,
) -> Any:

    # Check environment variable for cache setting (defaults to True)
    use_cache = os.getenv("USE_CACHE", "True").lower() in ("true", "1", "yes")
    original_messages = deepcopy(messages)
    for attempt in range(max_retries):
        messages = deepcopy(original_messages)
        if attempt > 0:
            messages[0]["content"] += f"\n\n(attempt no. {attempt + 1})"
        else:
            pass
        messages_json = make_hashable(messages)
        try:
            if use_cache:
                content = _cached_api_call(messages_json, model_id)
            else:
                content = _uncached_api_call(messages_json, model_id)
        except Exception as e:
            logger.error(f"API call failed on attempt {attempt+1}: {e}")
            continue

        logger.debug(f"Response on attempt {attempt+1}: type={type(content)}, content={repr(content[:200]) if content else 'None'}...")
        
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
