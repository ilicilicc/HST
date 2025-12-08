import time
import torch
import numpy as np
import importlib
import inspect

# A function to benchmark a given HST model
def benchmark_model(model_class, model_name, model_args):
    print(f"Benchmarking {model_name}...")

    # Instantiate the model
    try:
        model = model_class(**model_args)
    except Exception as e:
        print(f"Could not instantiate {model_name}: {e}")
        return 0

    # Create dummy input data
    batch_size = 32
    sequence_length = 100

    # Warm-up run and performance measurement
    try:
        X = torch.randint(0, model_args.get("vocab_size", 50257), (batch_size, sequence_length))

        # Warm-up
        _ = model(X)

        num_runs = 10
        start_time = time.time()
        for _ in range(num_runs):
            _ = model(X)
        end_time = time.time()

        total_tokens = batch_size * sequence_length * num_runs
        total_time = end_time - start_time
        tps = total_tokens / total_time

        print(f"{model_name} TPS: {tps:.2f}")
        return tps

    except Exception as e:
        print(f"Forward pass failed for {model_name}: {e}")
        return 0

# Find the correct class name in a module
def find_hst_class(module):
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and ("HST" in name or "Harmonic" in name):
            return obj
    return None

# Model configurations
model_configs = {
    "hst_v3": {"vocab_size": 50257, "d_model": 64, "n_heads": 4, "n_layers": 2},
    "hst_v4": {"vocab_size": 50257, "d_model": 64, "n_heads": 4, "n_layers": 2},
    "hst_v4_1": {"vocab_size": 50257, "d_model": 64, "n_heads": 4, "n_layers": 2},
    "hst_v5": {"vocab_size": 50257, "d_model": 64, "n_heads": 4, "n_layers": 2},
    "hst_v6": {"vocab_size": 50257, "d_model": 64, "n_heads": 4, "n_layers": 2},
    "hst_v6_1": {"vocab_size": 50257, "d_model": 64, "n_heads": 4, "n_layers": 2},
    "hst_v7_1": {"vocab_size": 50257, "d_model": 64, "n_heads": 4, "n_layers": 2},
    "hst_v7_1_2": {"vocab_size": 50257, "d_model": 64, "n_heads": 4, "n_layers": 2},
    "hst_v7_2": {"vocab_size": 50257, "d_model": 64, "n_heads": 4, "n_layers": 2},
    "hst_v8": {"vocab_size": 50257, "d_model": 64, "n_heads": 4, "n_layers": 2},
    "hst_v8_1": {"vocab_size": 50257, "d_model": 64, "n_heads": 4, "n_layers": 2},
    "hst_v8_2": {
        "vocab_size": 50257,
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 2,
    },
}

# Run the benchmark
if __name__ == "__main__":
    results = {}
    for model_name, args in model_configs.items():
        try:
            module = importlib.import_module(f"src.{model_name}")
            model_class = find_hst_class(module)
            if model_class:
                results[model_name] = benchmark_model(model_class, model_name, args)
            else:
                print(f"Could not find HST class in {model_name}")
        except ImportError as e:
            print(f"Could not import {model_name}: {e}")

    if results:
        # Filter out failed benchmarks
        valid_results = {k: v for k, v in results.items() if v > 0}
        if valid_results:
            # Sort the results by TPS in descending order
            sorted_results = sorted(valid_results.items(), key=lambda item: item[1], reverse=True)

            print("\n--- Benchmark Results ---")
            for model_name, tps in sorted_results:
                print(f"{model_name}: {tps:.2f} TPS")

            fastest_model = sorted_results[0][0]
            print(f"\nFastest model: {fastest_model} with {sorted_results[0][1]:.2f} TPS")

            # Save the fastest model name to a file
            with open("fastest_model.txt", "w") as f:
                f.write(fastest_model)
        else:
            print("\nNo models benchmarked successfully.")
