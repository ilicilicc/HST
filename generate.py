import time
import torch
import numpy as np
import importlib
import inspect

# Find the correct class name in a module
def find_hst_class(module):
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and ("HST" in name or "Harmonic" in name):
            return obj
    return None

def main():
    # Read the fastest model from the file
    try:
        with open("fastest_model.txt", "r") as f:
            fastest_model_name = f.read().strip()
    except FileNotFoundError:
        print("Could not find fastest_model.txt. Please run benchmark.py first.")
        return

    # Dynamically import the fastest model
    try:
        module = importlib.import_module(f"src.{fastest_model_name}")
        model_class = find_hst_class(module)
        if not model_class:
            print(f"Could not find HST class in {fastest_model_name}")
            return
    except ImportError:
        print(f"Could not import {fastest_model_name}")
        return

    # Model configuration
    model_args = {
        "vocab_size": 50257,
        "d_model": 256,
        "n_heads": 4,
        "n_layers": 8,
    }

    # Instantiate the model
    try:
        model = model_class(**model_args)
    except Exception as e:
        print(f"Could not instantiate {fastest_model_name}: {e}")
        return

    # Use a generic prompt
    prompt = "This is a generic prompt for text generation."

    # Convert the prompt to token IDs
    prompt_ids = torch.randint(0, model_args["vocab_size"], (1, len(prompt.split())))

    # Generate text
    print(f"Generating text with {fastest_model_name}...")
    start_time = time.time()
    try:
        # Use generate_ultra_fast if available, otherwise use generate
        if hasattr(model, "generate_ultra_fast"):
            output = model.generate_ultra_fast(prompt_ids, max_new_tokens=100)
        else:
            output = model.generate(prompt_ids, max_new_tokens=100)
    except Exception as e:
        print(f"Error during text generation: {e}")
        return
    end_time = time.time()

    # Handle different output formats from generate methods
    if isinstance(output, tuple):
        generated_ids = output[0]
    else:
        generated_ids = output

    # Calculate tokens per second (TPS)
    total_tokens = generated_ids.shape[1] - prompt_ids.shape[1]
    total_time = end_time - start_time
    tps = total_tokens / total_time

    print(f"\nText generation complete.")
    # a real implementation would convert the output IDs back to text
    print(f"Generated text (IDs): {generated_ids}")
    print(f"TPS: {tps:.2f}")

if __name__ == "__main__":
    main()
