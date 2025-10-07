#!/usr/bin/env python3

import torch
import yaml
import argparse
from calflops import calculate_flops
from nablafx.core.models import GreyBoxModel
from nablafx.processors import *


def create_model_from_yaml(yaml_path):
    """Create model from YAML config"""
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    model_config = config["model"]["init_args"]["model"]
    model_class = model_config["class_path"]
    model_args = model_config["init_args"]

    if "GreyBoxModel" in model_class:
        return GreyBoxModel(**model_args)

    elif "BlackBoxModel" in model_class:
        # Create black-box model directly
        proc_config = model_args["processor"]
        proc_class = proc_config["class_path"].split(".")[-1]
        proc_args = proc_config["init_args"]
        return globals()[proc_class](**proc_args)

    else:
        raise ValueError(f"Unknown model class: {model_class}")


def calculate_flops_from_yaml(yaml_path, seconds=1, sample_rate=48000):
    """Calculate FLOPs from YAML config"""
    print(f"Loading model from: {yaml_path}")

    # Create model
    model = create_model_from_yaml(yaml_path)
    print(f"Model created: {type(model).__name__}")

    # Create input tensors
    x = torch.rand(1, 1, sample_rate * seconds)

    # Check if model has controls
    num_controls = getattr(model, "num_controls", 0)
    print(f"Number of controls: {num_controls}")

    # Different models use different parameter names
    model_name = type(model).__name__
    if model_name in ["TCN", "GCN", "S4", "LSTM"]:  # Black-box models
        if num_controls > 0:
            controls = torch.rand(1, num_controls)
            kwargs = {"x": x, "p": controls}  # Black-box models use 'p' parameter
        else:
            kwargs = {"x": x}
    else:  # GreyBoxModel
        if num_controls > 0:
            controls = torch.rand(1, num_controls)
            kwargs = {"x": x, "controls": controls}  # GreyBox uses 'controls' parameter
        else:
            kwargs = {"x": x}

    print(f"Input shape: {x.shape}")
    print(f"Audio duration: {seconds} seconds")

    # Calculate FLOPs
    print("Calculating FLOPs...")
    flops, macs, params = calculate_flops(model=model, input_shape=None, kwargs=kwargs, output_as_string=True, output_precision=4)

    print(f"\nResults:")
    print(f"FLOPs: {flops}")
    print(f"MACs: {macs}")
    print(f"Parameters: {params}")

    return flops, macs, params


def main():
    parser = argparse.ArgumentParser(description="Calculate FLOPs from YAML model config")
    parser.add_argument("yaml_path", help="Path to YAML configuration file")
    parser.add_argument("--seconds", type=float, default=1.0, help="Audio duration in seconds")
    parser.add_argument("--sample-rate", type=int, default=48000, help="Sample rate")

    args = parser.parse_args()

    try:
        calculate_flops_from_yaml(args.yaml_path, args.seconds, args.sample_rate)
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
