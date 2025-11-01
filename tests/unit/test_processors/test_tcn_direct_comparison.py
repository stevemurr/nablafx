import torch
import sys
from pathlib import Path

# Add paths for standalone imports
standalone_path = Path(__file__).parent / "nablafx_tcn"
microtcn_path = Path(__file__).parent / "csteinmetz1_microtcn"
sys.path.insert(0, str(standalone_path))
sys.path.insert(0, str(microtcn_path))

from nablafx_tcn.tcn import TCN as NablAFxTCN
from csteinmetz1_microtcn.tcn import TCNModel as ReferenceTCN


def copy_weights_ref_to_nablafx(ref_model, nablafx_model, verbose=True):
    """
    Copy weights from reference model to NablAFx model.
    Maps corresponding layers between the two implementations.
    """
    if verbose:
        print("\n" + "=" * 80)
        print("COPYING WEIGHTS FROM REFERENCE TO NABLAFX")
        print("=" * 80)

    copied_layers = []
    failed_layers = []

    # 1. Copy conditioning MLP weights (gen -> cond_nn)
    try:
        # Reference: gen = Sequential(Linear, ReLU, Linear, ReLU, Linear, ReLU)
        # NablAFx: cond_nn = MLP with layers = [Linear, ReLU, Linear, ReLU, Linear, ReLU]
        ref_linear_layers = [ref_model.gen[i] for i in [0, 2, 4]]  # indices of Linear layers
        nablafx_linear_layers = [nablafx_model.cond_nn.layers[i] for i in [0, 2, 4]]  # indices of Linear layers

        for i, (ref_layer, nablafx_layer) in enumerate(zip(ref_linear_layers, nablafx_linear_layers)):
            nablafx_layer.weight.data.copy_(ref_layer.weight.data)
            nablafx_layer.bias.data.copy_(ref_layer.bias.data)

        copied_layers.append("Conditioning MLP (3 Linear layers)")
    except Exception as e:
        failed_layers.append(f"Conditioning MLP: {e}")

    # 2. Copy TCN blocks
    num_blocks = len(ref_model.blocks)
    for block_idx in range(num_blocks):
        ref_block = ref_model.blocks[block_idx]
        nablafx_block = nablafx_model.blocks[block_idx]

        try:
            # Conv1d weights (conv1 -> conv)
            nablafx_block.conv.weight.data.copy_(ref_block.conv1.weight.data)
            copied_layers.append(f"Block {block_idx}: Conv1d")
        except Exception as e:
            failed_layers.append(f"Block {block_idx} Conv1d: {e}")

        try:
            # FiLM layer weights
            # Reference: film.bn (no affine), film.adaptor
            # NablAFx: film.bn (no affine), film.adaptor

            # BatchNorm - running stats only (no learnable params due to affine=False)
            nablafx_block.film.bn.running_mean.data.copy_(ref_block.film.bn.running_mean.data)
            nablafx_block.film.bn.running_var.data.copy_(ref_block.film.bn.running_var.data)
            nablafx_block.film.bn.num_batches_tracked.data.copy_(ref_block.film.bn.num_batches_tracked.data)

            # FiLM adaptor (Linear layer)
            nablafx_block.film.adaptor.weight.data.copy_(ref_block.film.adaptor.weight.data)
            nablafx_block.film.adaptor.bias.data.copy_(ref_block.film.adaptor.bias.data)

            copied_layers.append(f"Block {block_idx}: FiLM layer")
        except Exception as e:
            failed_layers.append(f"Block {block_idx} FiLM: {e}")

        try:
            # PReLU activation (act -> relu)
            nablafx_block.act.weight.data.copy_(ref_block.relu.weight.data)
            copied_layers.append(f"Block {block_idx}: PReLU")
        except Exception as e:
            failed_layers.append(f"Block {block_idx} PReLU: {e}")

        try:
            # Residual connection (res -> res)
            nablafx_block.res.weight.data.copy_(ref_block.res.weight.data)
            copied_layers.append(f"Block {block_idx}: Residual")
        except Exception as e:
            failed_layers.append(f"Block {block_idx} Residual: {e}")

    # 3. Copy output layer
    try:
        nablafx_model.output.weight.data.copy_(ref_model.output.weight.data)
        # Only copy bias if both models have it
        if hasattr(nablafx_model.output, "bias") and nablafx_model.output.bias is not None:
            if hasattr(ref_model.output, "bias") and ref_model.output.bias is not None:
                nablafx_model.output.bias.data.copy_(ref_model.output.bias.data)
                copied_layers.append("Output layer (weight + bias)")
            else:
                copied_layers.append("Output layer (weight only, ref has no bias)")
        else:
            copied_layers.append("Output layer (weight only, nablafx has no bias)")
    except Exception as e:
        failed_layers.append(f"Output layer: {e}")

    # Summary
    if verbose:
        print(f"\n✅ Successfully copied {len(copied_layers)} layer groups:")
        for layer in copied_layers[:5]:  # Show first 5
            print(f"   - {layer}")
        if len(copied_layers) > 5:
            print(f"   ... and {len(copied_layers) - 5} more")

        if failed_layers:
            print(f"\n⚠️  Failed to copy {len(failed_layers)} layer groups:")
            for layer in failed_layers:
                print(f"   - {layer}")

        print("=" * 80 + "\n")

    return len(failed_layers) == 0


def test_identical_weights_identical_output():
    """
    Initialize both models with same weights and verify outputs match
    """
    print("\n" + "=" * 80)
    print("TEST: Identical Weights, Identical Output")
    print("=" * 80)

    # Configuration
    config = {
        "num_inputs": 1,
        "num_outputs": 1,
        "num_params": 3,
        "num_blocks": 4,
        "kernel_size": 5,
        "dilation_growth": 10,
        "channel_growth": 1,
        "channel_width": 32,
        "stack_size": 10,
        "causal": True,
    }

    # Create both models
    ref_model = ReferenceTCN(
        ninputs=config["num_inputs"],
        noutputs=config["num_outputs"],
        nparams=config["num_params"],
        nblocks=config["num_blocks"],
        kernel_size=config["kernel_size"],
        dilation_growth=config["dilation_growth"],
        channel_growth=config["channel_growth"],
        channel_width=config["channel_width"],
        stack_size=config["stack_size"],
        grouped=False,
        causal=config["causal"],
        skip_connections=False,  # skip connections not implemented in reference model
        num_examples=0,
    )

    nablafx_model = NablAFxTCN(
        num_inputs=config["num_inputs"],
        num_outputs=config["num_outputs"],
        num_controls=config["num_params"],
        num_blocks=config["num_blocks"],
        kernel_size=config["kernel_size"],
        dilation_growth=config["dilation_growth"],
        channel_growth=config["channel_growth"],
        channel_width=config["channel_width"],
        stack_size=config["stack_size"],
        groups=1,  # group convolutions disabled to match reference implementation
        bias=False,  # bias disabled to match reference implementation
        causal=config["causal"],
        batchnorm=False,  # batchnorm only active for unconditioned blocks
        residual=True,  # residual connections to match reference implementation
        direct_path=False,  # no direct path
        cond_type="film",  # FiLM conditioning to match reference implementation
        cond_block_size=128,  # not used for FiLM
        cond_num_layers=1,  # not used for FiLM
        act_type="prelu",  # PReLU activation to match reference implementation
    )

    # Initialize reference model with random seed
    torch.manual_seed(42)
    ref_model.apply(lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None)

    # Copy weights from reference to NablAFx model
    copy_success = copy_weights_ref_to_nablafx(ref_model, nablafx_model)

    # Verify weights were copied correctly
    print("-" * 80)
    print("WEIGHT VERIFICATION AFTER COPYING")
    print("-" * 80)

    # Count parameters
    ref_params = sum(p.numel() for p in ref_model.parameters())
    nablafx_params = sum(p.numel() for p in nablafx_model.parameters())
    print(f"Reference model parameters: {ref_params:,}")
    print(f"NablAFx model parameters: {nablafx_params:,}")

    # Check first conv layer weights
    try:
        ref_first_conv = ref_model.blocks[0].conv1.weight
        nablafx_first_conv = nablafx_model.blocks[0].conv.weight
        print(f"\nFirst conv weight shapes:")
        print(f"  Reference: {ref_first_conv.shape}")
        print(f"  NablAFx: {nablafx_first_conv.shape}")

        if ref_first_conv.shape == nablafx_first_conv.shape:
            weight_diff = torch.abs(ref_first_conv - nablafx_first_conv).mean().item()
            print(f"  Mean absolute difference: {weight_diff:.10f}")
            if weight_diff < 1e-6:
                print(f"  ✅ First conv weights are identical")
            else:
                print(f"  ⚠️  First conv weights differ: {weight_diff:.10f}")
        else:
            print(f"  ⚠️  Shapes don't match - architectures differ")
    except Exception as e:
        print(f"  ❌ Could not compare conv weights: {e}")

    # Check conditioning MLP weights
    try:
        ref_mlp_weight = ref_model.gen[0].weight  # First Linear layer
        nablafx_mlp_weight = nablafx_model.cond_nn.layers[0].weight  # First Linear layer
        print(f"\nConditioning MLP first layer weight shapes:")
        print(f"  Reference: {ref_mlp_weight.shape}")
        print(f"  NablAFx: {nablafx_mlp_weight.shape}")

        if ref_mlp_weight.shape == nablafx_mlp_weight.shape:
            mlp_diff = torch.abs(ref_mlp_weight - nablafx_mlp_weight).mean().item()
            print(f"  Mean absolute difference: {mlp_diff:.10f}")
            if mlp_diff < 1e-6:
                print(f"  ✅ MLP weights are identical")
            else:
                print(f"  ⚠️  MLP weights differ: {mlp_diff:.10f}")
        else:
            print(f"  ⚠️  Shapes don't match")
    except Exception as e:
        print(f"  ❌ Could not compare MLP weights: {e}")

    print("-" * 80)

    # Set to eval mode
    ref_model.eval()
    nablafx_model.eval()

    # Create random input and parameters
    torch.manual_seed(123)
    batch_size = 2
    seq_length = 16000
    x = torch.randn(batch_size, 1, seq_length)
    if config["causal"]:
        x_ref = torch.nn.functional.pad(x, (ref_model.compute_receptive_field() - 1, 0))
    else:
        x_ref = torch.nn.functional.pad(x, ((ref_model.compute_receptive_field() - 1) // 2, (ref_model.compute_receptive_field() - 1) // 2))
    x_nablafx = x.clone()  # nablafx model handles padding internally

    # Both models expect (batch, params) - static parameters
    p_ref = torch.randn(batch_size, 1, config["num_params"])
    p_nablafx = p_ref.squeeze(1)  # nablafx expects (batch, params)

    print(f"\nInput shapes:")
    print(f"  Audio: {x.shape}")
    print(f"  Audio (Ref): {x_ref.shape}")
    print(f"  Audio (NablAFx): {x_nablafx.shape} - NablAFx handles padding internally")
    print(f"  Params Ref: {p_ref.shape}")
    print(f"  Params NablAFx: {p_nablafx.shape}")

    # Verify parameters are identical
    print(f"\nParameter values (first sample):")
    print(f"  Ref: {p_ref[0, 0, :].tolist()}")
    print(f"  NablAFx: {p_nablafx[0, :].tolist()}")
    params_match = torch.allclose(p_ref.squeeze(1), p_nablafx)
    print(f"  Parameters match: {params_match}")

    # Forward pass
    with torch.no_grad():
        try:
            y_ref = ref_model(x_ref, p_ref)
            print(f"\n✅ Reference model forward pass successful")
            print(f"   Output shape: {y_ref.shape}")
            print(f"   Output range: [{y_ref.min():.4f}, {y_ref.max():.4f}]")
        except Exception as e:
            print(f"\n❌ Reference model failed: {e}")
            raise

        try:
            y_nablafx = nablafx_model(x_nablafx, p_nablafx)
            print(f"\n✅ NablAFx model forward pass successful")
            print(f"   Output shape: {y_nablafx.shape}")
            print(f"   Output range: [{y_nablafx.min():.4f}, {y_nablafx.max():.4f}]")
        except Exception as e:
            print(f"\n❌ NablAFx model failed: {e}")
            raise

    # Check shapes match
    print(f"\nShape comparison:")
    print(f"  Batch: ref={y_ref.shape[0]}, nablafx={y_nablafx.shape[0]}")
    print(f"  Channels: ref={y_ref.shape[1]}, nablafx={y_nablafx.shape[1]}")
    print(f"  Time: ref={y_ref.shape[2]}, nablafx={y_nablafx.shape[2]}")

    assert y_ref.shape[0] == y_nablafx.shape[0], "Batch size mismatch"
    assert y_ref.shape[1] == y_nablafx.shape[1], "Channel count mismatch"

    # Compare outputs (crop to minimum length)
    min_length = min(y_ref.shape[2], y_nablafx.shape[2])
    y_ref_crop = y_ref[:, :, :min_length]
    y_nablafx_crop = y_nablafx[:, :, :min_length]

    abs_diff = torch.abs(y_ref_crop - y_nablafx_crop)
    mean_abs_diff = abs_diff.mean().item()
    max_abs_diff = abs_diff.max().item()

    print(f"\nNumerical comparison (cropped to {min_length} samples):")
    print(f"  Mean absolute difference: {mean_abs_diff:.6f}")
    print(f"  Max absolute difference: {max_abs_diff:.6f}")

    # Correlation check
    correlation = torch.corrcoef(torch.stack([y_ref_crop.flatten(), y_nablafx_crop.flatten()]))[0, 1]
    print(f"  Output correlation: {correlation:.6f}")

    # Results
    print("\n" + "-" * 80)
    if mean_abs_diff < 1e-5 and correlation > 0.99:
        print("✅ TEST PASSED: Implementations match!")
        print("   Both models produce nearly identical outputs")
    elif correlation > 0.95:
        print("⚠️  TEST PARTIAL: Implementations are similar but not identical")
        print(f"   Mean diff: {mean_abs_diff:.6f} (threshold: 1e-5)")
        print(f"   Correlation: {correlation:.6f} (threshold: 0.99)")
        print("   This may be due to:")
        print("   - Different initialization strategies")
        print("   - Different parameter conditioning methods")
        print("   - Architectural differences in FiLM implementation")
    else:
        print("❌ TEST FAILED: Implementations differ significantly")
        print(f"   Mean diff: {mean_abs_diff:.6f} (threshold: 1e-5)")
        print(f"   Correlation: {correlation:.6f} (threshold: 0.95)")


def test_deterministic():
    """Test both models are deterministic - i.e. same input and params yield same output on repeated calls"""
    print("\n" + "=" * 80)
    print("TEST: Determinism")
    print("=" * 80)

    model = NablAFxTCN(num_controls=3, num_blocks=2, channel_width=16, cond_type="film")
    model.eval()

    x = torch.randn(1, 1, 1000)
    p = torch.randn(1, 3)

    with torch.no_grad():
        y1 = model(x, p)
        y2 = model(x, p)

    diff = torch.abs(y1 - y2).max().item()

    if diff == 0:
        print("✅ TEST PASSED: Model is deterministic")
    else:
        print(f"⚠️  TEST FAILED: Model is non-deterministic (diff: {diff})")


def test_multiple_configurations():
    """
    Test multiple configurations with copied weights.
    Verifies both implementations work correctly and produce identical outputs
    across different architectural configurations.
    """
    print("\n" + "=" * 80)
    print("TEST: Multiple Configurations")
    print("=" * 80)

    configs = [
        (2, 3, 2, True),
        (4, 3, 5, False),
        (8, 5, 10, True),
    ]

    for num_blocks, kernel_size, dilation_growth, causal in configs:
        print(f"\nTesting: blocks={num_blocks}, kernel={kernel_size}, dilation={dilation_growth}, causal={causal}")

        config = {
            "num_inputs": 1,
            "num_outputs": 1,
            "num_params": 3,
            "num_blocks": num_blocks,
            "kernel_size": kernel_size,
            "dilation_growth": dilation_growth,
            "channel_growth": 1,
            "channel_width": 16,
            "stack_size": 4,
            "causal": causal,
        }

        try:
            # Set seed BEFORE creating models to ensure consistent initialization
            torch.manual_seed(42)

            ref_model = ReferenceTCN(
                ninputs=config["num_inputs"],
                noutputs=config["num_outputs"],
                nparams=config["num_params"],
                nblocks=config["num_blocks"],
                kernel_size=config["kernel_size"],
                dilation_growth=config["dilation_growth"],
                channel_growth=config["channel_growth"],
                channel_width=config["channel_width"],
                stack_size=config["stack_size"],
                grouped=False,
                causal=config["causal"],
                skip_connections=False,  # skip connections not implemented in reference model
                num_examples=0,
            )

            nablafx_model = NablAFxTCN(
                num_inputs=config["num_inputs"],
                num_outputs=config["num_outputs"],
                num_controls=config["num_params"],
                num_blocks=config["num_blocks"],
                kernel_size=config["kernel_size"],
                dilation_growth=config["dilation_growth"],
                channel_growth=config["channel_growth"],
                channel_width=config["channel_width"],
                stack_size=config["stack_size"],
                groups=1,  # group convolutions disabled to match reference implementation
                bias=False,  # bias disabled to match reference implementation
                causal=config["causal"],
                batchnorm=False,  # batchnorm only active for unconditioned blocks
                residual=True,  # residual connections to match reference implementation
                direct_path=False,  # no direct path
                cond_type="film",  # FiLM conditioning to match reference implementation
                cond_block_size=128,  # not used for FiLM
                cond_num_layers=1,  # not used for FiLM
                act_type="prelu",  # PReLU activation to match reference implementation
            )

            # Re-initialize reference model with fixed seed and copy weights to NablAFx
            torch.manual_seed(42)
            ref_model.apply(lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None)
            success = copy_weights_ref_to_nablafx(ref_model, nablafx_model, verbose=False)
            if not success:
                print(f"    ⚠️ Weight copying failed!")
                continue

            ref_model.eval()
            nablafx_model.eval()

            # Use fixed seed for reproducible test inputs
            torch.manual_seed(123)
            x = torch.randn(1, 1, 16000)

            # Reference model needs manual padding (NablAFx handles it internally)
            if config["causal"]:
                x_ref = torch.nn.functional.pad(x, (ref_model.compute_receptive_field() - 1, 0))
            else:
                rf = ref_model.compute_receptive_field()
                x_ref = torch.nn.functional.pad(x, ((rf - 1) // 2, (rf - 1) // 2))

            x_nablafx = x.clone()  # NablAFx handles padding internally

            p_ref = torch.randn(1, 1, config["num_params"])  # Reference expects [batch, 1, params]
            p_nablafx = p_ref.squeeze(1)  # NablAFx expects [batch, params]

            with torch.no_grad():
                y_ref = ref_model(x_ref, p_ref)
                y_nablafx = nablafx_model(x_nablafx, p_nablafx)

            min_len = min(y_ref.shape[2], y_nablafx.shape[2])

            # Compare outputs with copied weights
            abs_diff = torch.abs(y_ref[0, 0, :min_len] - y_nablafx[0, 0, :min_len]).mean().item()
            correlation = torch.corrcoef(torch.stack([y_ref[0, 0, :min_len], y_nablafx[0, 0, :min_len]]))[0, 1].item()

            # With copied weights, outputs should match exactly
            if abs_diff < 1e-5 and correlation > 0.999:
                status = "✅"
            else:
                status = "⚠️"
            print(f"  {status} Mean diff: {abs_diff:.6f}, Correlation: {correlation:.6f}")

        except Exception as e:
            print(f"  ❌ Configuration failed: {e}")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("TCN IMPLEMENTATION VERIFICATION")
    print("=" * 80)

    try:
        test_identical_weights_identical_output()
        test_deterministic()
        test_multiple_configurations()

        print("\n" + "=" * 80)
        print("ALL TESTS COMPLETED")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
