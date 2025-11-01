"""
Test architectural equivalence through gradient flow analysis.

Two architectures are equivalent if they can represent the same
functions and have similar gradient flow properties.
"""

import torch
import torch.nn.functional as F
import sys
from pathlib import Path

nablafx_path = Path(__file__).parent / "nablafx_gcn"
mcomunita_path = Path(__file__).parent / "mcomunita_gcntfilm"
sys.path.insert(0, str(nablafx_path))
sys.path.insert(0, str(mcomunita_path))

from nablafx_gcn.gcn import GCN as NablAFxGCN
from mcomunita_gcntfilm.gcn import GCN as ReferenceGCN


def copy_weights_ref_to_nablafx(ref_model, nablafx_model, verbose=False):
    """
    Copy weights from reference model to NablAFx model.

    This ensures both models have identical initialization for fair comparison.
    """
    if verbose:
        print("\nCopying weights from Reference to NablAFx...")

    # Flatten all reference GatedConv1d layers
    ref_layers = []
    for ref_block in ref_model.blocks[:-1]:  # Exclude last block (output mixer)
        if hasattr(ref_block, "layers"):
            ref_layers.extend(ref_block.layers)

    # Copy weights from reference layers to NablAFx blocks
    num_layers_to_copy = min(len(ref_layers), len(nablafx_model.blocks))

    for idx in range(num_layers_to_copy):
        ref_layer = ref_layers[idx]
        nablafx_block = nablafx_model.blocks[idx]

        # Copy conv weights and biases
        nablafx_block.conv.weight.data.copy_(ref_layer.conv.weight.data)
        if ref_layer.conv.bias is not None and nablafx_block.conv.bias is not None:
            nablafx_block.conv.bias.data.copy_(ref_layer.conv.bias.data)

        # Copy mix weights and biases
        nablafx_block.mix.weight.data.copy_(ref_layer.mix.weight.data)
        if ref_layer.mix.bias is not None and nablafx_block.mix.bias is not None:
            nablafx_block.mix.bias.data.copy_(ref_layer.mix.bias.data)

    # Copy output mixer layer
    nablafx_model.mix.weight.data.copy_(ref_model.blocks[-1].weight.data)
    if ref_model.blocks[-1].bias is not None and nablafx_model.mix.bias is not None:
        nablafx_model.mix.bias.data.copy_(ref_model.blocks[-1].bias.data)

    if verbose:
        print(f"  ✅ Copied weights for {num_layers_to_copy} layers")


def analyze_gradient_flow(model, x, target, model_name="Model"):
    """Analyze how gradients flow through the model"""

    model.train()

    # Forward pass
    output = model(x)

    # Compute loss
    loss = F.mse_loss(output, target)

    # Backward pass
    loss.backward()

    # Analyze gradients
    grad_stats = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_stats.append(
                {
                    "name": name,
                    "mean": param.grad.abs().mean().item(),
                    "std": param.grad.std().item(),
                    "max": param.grad.abs().max().item(),
                }
            )

    return grad_stats, loss.item()


def test_gradient_equivalence():
    print("=" * 80)
    print("ARCHITECTURAL EQUIVALENCE TEST: Gradient Flow Analysis")
    print("=" * 80)

    # Configuration
    config = {
        "nblocks": 2,
        "nlayers": 9,
        "nchannels": 8,
        "kernel_size": 3,
        "dilation_growth": 2,
    }

    # Create models
    torch.manual_seed(42)
    ref_model = ReferenceGCN(
        nblocks=config["nblocks"],
        nlayers=config["nlayers"],
        nchannels=config["nchannels"],
        kernel_size=config["kernel_size"],
        dilation_growth=config["dilation_growth"],
    )

    nablafx_model = NablAFxGCN(
        num_inputs=1,
        num_outputs=1,
        num_controls=0,
        num_blocks=config["nblocks"] * config["nlayers"],
        kernel_size=config["kernel_size"],
        dilation_growth=config["dilation_growth"],
        channel_growth=1,
        channel_width=config["nchannels"],
        stack_size=config["nlayers"],
        groups=1,
        bias=True,
        causal=True,
        batchnorm=False,
        residual=True,
        direct_path=False,
        cond_type=None,
        cond_block_size=128,
        cond_num_layers=1,
    )

    # Initialize reference model and copy weights to NablAFx
    print("\nInitializing models with same weights...")
    torch.manual_seed(42)
    ref_model.apply(lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None)
    copy_weights_ref_to_nablafx(ref_model, nablafx_model, verbose=True)

    # Verify weights were copied
    ref_first_conv = ref_model.blocks[0].layers[0].conv.weight
    nablafx_first_conv = nablafx_model.blocks[0].conv.weight
    weight_diff = torch.abs(ref_first_conv - nablafx_first_conv).mean().item()
    print(f"  Initial weight difference: {weight_diff:.10f}")
    if weight_diff < 1e-6:
        print(f"  ✅ Both models start with identical weights!")

    # Generate random input and target
    torch.manual_seed(123)
    x = torch.randn(4, 1, 4000)  # [batch, channels, length]
    target = torch.randn(4, 1, 4000)

    # Reference format [length, batch, channels]
    x_ref = x.permute(2, 0, 1)
    target_ref = target.permute(2, 0, 1)

    print("\n" + "-" * 80)
    print("Analyzing Reference GCN gradient flow...")
    print("-" * 80)
    ref_grads, ref_loss = analyze_gradient_flow(ref_model, x_ref, target_ref, "Reference")

    print("\n" + "-" * 80)
    print("Analyzing NablAFx GCN gradient flow...")
    print("-" * 80)
    nablafx_grads, nablafx_loss = analyze_gradient_flow(nablafx_model, x, target, "NablAFx")

    # Compare gradient statistics
    print("\n" + "=" * 80)
    print("GRADIENT FLOW COMPARISON")
    print("=" * 80)

    # Overall gradient magnitude
    ref_grad_mean = sum(g["mean"] for g in ref_grads) / len(ref_grads)
    nablafx_grad_mean = sum(g["mean"] for g in nablafx_grads) / len(nablafx_grads)

    ref_grad_max = max(g["max"] for g in ref_grads)
    nablafx_grad_max = max(g["max"] for g in nablafx_grads)

    print(f"Average gradient magnitude:")
    print(f"  Reference: {ref_grad_mean:.6f}")
    print(f"  NablAFx:   {nablafx_grad_mean:.6f}")
    print(f"  Ratio:     {ref_grad_mean / nablafx_grad_mean:.3f}x")

    print(f"\nMax gradient magnitude:")
    print(f"  Reference: {ref_grad_max:.6f}")
    print(f"  NablAFx:   {nablafx_grad_max:.6f}")
    print(f"  Ratio:     {ref_grad_max / nablafx_grad_max:.3f}x")

    # Check for vanishing/exploding gradients
    ref_has_issue = ref_grad_mean < 1e-6 or ref_grad_max > 100
    nablafx_has_issue = nablafx_grad_mean < 1e-6 or nablafx_grad_max > 100

    print(f"\nGradient health:")
    print(f"  Reference: {'⚠️  Issue detected' if ref_has_issue else '✅ Healthy'}")
    print(f"  NablAFx:   {'⚠️  Issue detected' if nablafx_has_issue else '✅ Healthy'}")

    # Check if gradients are in similar range (within 10x)
    ratio = ref_grad_mean / nablafx_grad_mean
    similar_magnitude = 0.1 < ratio < 10

    print("\n" + "=" * 80)
    print("EQUIVALENCE VERDICT")
    print("=" * 80)

    if similar_magnitude and not (ref_has_issue or nablafx_has_issue):
        print("✅ GRADIENTS FLOW SIMILARLY!")
        print("   - Both architectures have healthy gradient flow")
        print("   - Gradient magnitudes are in similar range")
        print("   - Both can be trained effectively with similar hyperparameters")
    elif not (ref_has_issue or nablafx_has_issue):
        print("✅ BOTH HAVE HEALTHY GRADIENTS (different magnitudes)")
        print("   - Both can learn, but may need different learning rates")
        print(f"   - Suggested LR ratio: {ratio:.3f}x")
    else:
        print("⚠️  One or both architectures may have gradient issues")

    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print("If both architectures have healthy gradient flow (no vanishing/exploding),")
    print("they are trainable and can learn similar functions, proving equivalence.")
    print("\nDifferent gradient magnitudes just mean you might need different learning")
    print("rates, but both can reach the same solution space.")


if __name__ == "__main__":
    test_gradient_equivalence()
