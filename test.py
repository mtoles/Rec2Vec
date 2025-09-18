import time
import openai
from utils.retry import retry_with_fallback
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def simple_api_call(messages, model_id):
    """Simple API call without retry or JSON formatting."""
    response = openai.chat.completions.create(
        model=model_id,
        messages=messages,
    )
    return response.choices[0].message.content


def test_with_retry_and_json(prompts, model_id):
    """Test with retry mechanism and JSON formatting."""
    print("=" * 60)
    print("TESTING WITH RETRY AND JSON FORMATTING")
    print("=" * 60)

    results = []
    for i, prompt in enumerate(prompts):
        messages = [{"role": "user", "content": prompt}]

        def always_true(_):
            return True  # Accept any response

        start = time.time()
        response = retry_with_fallback(
            messages=messages,
            validation_func=always_true,
            max_retries=3,
            fallback_value="No response",
            model_id=model_id,
        )
        elapsed = time.time() - start
        print(f"Prompt {i+1}: {prompt}")
        print(f"Response: {response}")
        print(f"Time taken: {elapsed:.3f} seconds\n")
        results.append((prompt, response, elapsed))

    return results


def test_without_retry_and_json(prompts, model_id):
    """Test without retry mechanism and JSON formatting."""
    print("=" * 60)
    print("TESTING WITHOUT RETRY AND JSON FORMATTING")
    print("=" * 60)

    # Remove JSON formatting from prompts
    simple_prompts = [
        "What is the capital of France?",
        "Summarize the theory of relativity in one sentence.",
        "List three uses for a paperclip.",
    ]

    results = []
    for i, prompt in enumerate(simple_prompts):
        messages = [{"role": "user", "content": prompt}]

        start = time.time()
        response = simple_api_call(messages, model_id)
        elapsed = time.time() - start
        print(f"Prompt {i+1}: {prompt}")
        print(f"Response: {response}")
        print(f"Time taken: {elapsed:.3f} seconds\n")
        results.append((prompt, response, elapsed))

    return results


def main():
    prompts = [
        "What is the capital of France? Please respond in JSON format.",
        "Summarize the theory of relativity in one sentence. Return your answer as JSON.",
        "List three uses for a paperclip. Format your response as JSON.",
    ]
    model_id = "gpt-5-nano"

    # Test with retry and JSON formatting
    retry_results = test_with_retry_and_json(prompts, model_id)

    # Test without retry and JSON formatting
    simple_results = test_without_retry_and_json(prompts, model_id)

    # Calculate and compare average latencies
    retry_times = [result[2] for result in retry_results]
    simple_times = [result[2] for result in simple_results]

    retry_avg = sum(retry_times) / len(retry_times)
    simple_avg = sum(simple_times) / len(simple_times)

    print("=" * 60)
    print("LATENCY COMPARISON")
    print("=" * 60)
    print(f"With retry + JSON formatting:")
    print(f"  Individual times: {[f'{t:.3f}s' for t in retry_times]}")
    print(f"  Average latency: {retry_avg:.3f} seconds")
    print()
    print(f"Without retry + JSON formatting:")
    print(f"  Individual times: {[f'{t:.3f}s' for t in simple_times]}")
    print(f"  Average latency: {simple_avg:.3f} seconds")
    print()
    print(f"Difference: {retry_avg - simple_avg:.3f} seconds")
    print(f"Speedup: {retry_avg / simple_avg:.2f}x faster without retry/JSON")
    print("=" * 60)


if __name__ == "__main__":
    main()
