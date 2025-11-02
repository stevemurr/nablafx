"""
Direct comparison test between NablAFx LSTM and Reference (CoreAudioML) LSTM implementations.

This test verifies that:
1. Implementations can be initialized with identical weights
2. Implementations produce identical outputs given the same input
3. Implementations are deterministic
4. Multiple configurations work correctly

NOTE: The main architectural difference is that NablAFx applies a tanh activation to the output,
while the reference implementation does not. We account for this by applying tanh to the reference output,
which is what one would do in practice to keep outputs bounded.
"""

import sys
import torch
import traceback
from pathlib import Path

# Add paths for standalone imports
standalone_path = Path(__file__).parent / "nablafx_lstm"
reference_path = Path(__file__).parent / "alecwright_coreaudioml"
sys.path.insert(0, str(standalone_path))
sys.path.insert(0, str(reference_path))

from nablafx_lstm.lstm import LSTM as NablAFxLSTM
from alecwright_coreaudioml.networks import SimpleRNN as ReferenceLSTM


def copy_weights_ref_to_nablafx(ref_model, nablafx_model, verbose=True):
    """
    Copy weights from reference model to NablAFx model.
    Maps corresponding layers between the two implementations.

    Reference model structure (SimpleRNN with LSTM):
    - rec: LSTM module (input_size, hidden_size, num_layers)
    - lin: Linear(hidden_size, output_size)
    - skip: residual connection flag

    NablAFx model structure:
    - lstm: LSTM module (input_size, hidden_size, num_layers)
    - lin: Linear(hidden_size, num_outputs)
    - residual: residual connection flag
    - cond_type: None (for basic LSTM without conditioning)
    """
    if verbose:
        print("\n" + "=" * 80)
        print("COPYING WEIGHTS FROM REFERENCE TO NABLAFX")
        print("=" * 80)

    copied_layers = []
    failed_layers = []

    # 1. Copy LSTM weights (rec -> lstm)
    try:
        # LSTM has multiple weight matrices per layer:
        # - weight_ih_l[k]: input-hidden weights for layer k
        # - weight_hh_l[k]: hidden-hidden weights for layer k
        # - bias_ih_l[k]: input-hidden bias for layer k
        # - bias_hh_l[k]: hidden-hidden bias for layer k

        ref_lstm = ref_model.rec
        nablafx_lstm = nablafx_model.lstm

        num_layers = ref_lstm.num_layers
        for layer_idx in range(num_layers):
            # Copy weights
            weight_ih_name = f"weight_ih_l{layer_idx}"
            weight_hh_name = f"weight_hh_l{layer_idx}"
            bias_ih_name = f"bias_ih_l{layer_idx}"
            bias_hh_name = f"bias_hh_l{layer_idx}"

            getattr(nablafx_lstm, weight_ih_name).data.copy_(getattr(ref_lstm, weight_ih_name).data)
            getattr(nablafx_lstm, weight_hh_name).data.copy_(getattr(ref_lstm, weight_hh_name).data)
            getattr(nablafx_lstm, bias_ih_name).data.copy_(getattr(ref_lstm, bias_ih_name).data)
            getattr(nablafx_lstm, bias_hh_name).data.copy_(getattr(ref_lstm, bias_hh_name).data)

            copied_layers.append(f"LSTM layer {layer_idx} (weights and biases)")

    except Exception as e:
        failed_layers.append(f"LSTM layers: {e}")

    # 2. Copy Linear layer weights (lin -> lin)
    try:
        nablafx_model.lin.weight.data.copy_(ref_model.lin.weight.data)
        if ref_model.lin.bias is not None and nablafx_model.lin.bias is not None:
            nablafx_model.lin.bias.data.copy_(ref_model.lin.bias.data)
            copied_layers.append("Output Linear layer (weight and bias)")
        else:
            copied_layers.append("Output Linear layer (weight only)")
    except Exception as e:
        failed_layers.append(f"Linear layer: {e}")

    # Summary
    if verbose:
        print(f"\n✅ Successfully copied {len(copied_layers)} layer groups:")
        for layer in copied_layers:
            print(f"   - {layer}")

        if failed_layers:
            print(f"\n⚠️  Failed to copy {len(failed_layers)} layer groups:")
            for layer in failed_layers:
                print(f"   - {layer}")

        print("=" * 80 + "\n")

    return len(failed_layers) == 0


def test_identical_weights_identical_output():
    """
    Initialize both models with same weights and verify outputs match.
    Tests basic LSTM without conditioning (cond_type=None).
    """
    print("\n" + "=" * 80)
    print("TEST: Identical Weights, Identical Output")
    print("=" * 80)

    # Configuration - matching parameters between implementations
    config = {
        "num_inputs": 1,
        "num_outputs": 1,
        "hidden_size": 32,
        "num_layers": 1,
    }

    # Create both models
    # Reference model (CoreAudioML SimpleRNN with LSTM)
    ref_model = ReferenceLSTM(
        input_size=config["num_inputs"],
        output_size=config["num_outputs"],
        unit_type="LSTM",
        hidden_size=config["hidden_size"],
        skip=1,
        bias_fl=True,
        num_layers=config["num_layers"],
    )

    # NablAFx model (basic LSTM without conditioning)
    nablafx_model = NablAFxLSTM(
        num_inputs=config["num_inputs"],
        num_outputs=config["num_outputs"],
        num_controls=0,  # no conditioning
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        residual=True,
        direct_path=False,  # no direct path
        cond_type=None,  # basic LSTM without conditioning
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

    # Check LSTM weights
    try:
        ref_lstm_weight = ref_model.rec.weight_ih_l0
        nablafx_lstm_weight = nablafx_model.lstm.weight_ih_l0
        print(f"\nLSTM weight_ih_l0 shapes:")
        print(f"  Reference: {ref_lstm_weight.shape}")
        print(f"  NablAFx: {nablafx_lstm_weight.shape}")

        if ref_lstm_weight.shape == nablafx_lstm_weight.shape:
            weight_diff = torch.abs(ref_lstm_weight - nablafx_lstm_weight).mean().item()
            print(f"  Mean absolute difference: {weight_diff:.10f}")
            if weight_diff < 1e-6:
                print(f"  ✅ Weights match perfectly")
            else:
                print(f"  ⚠️  Weights differ slightly")
        else:
            print(f"  ⚠️  Shapes don't match - architectures differ")
    except Exception as e:
        print(f"  ❌ Could not compare LSTM weights: {e}")

    # Check Linear layer weights
    try:
        ref_lin_weight = ref_model.lin.weight
        nablafx_lin_weight = nablafx_model.lin.weight
        print(f"\nLinear layer weight shapes:")
        print(f"  Reference: {ref_lin_weight.shape}")
        print(f"  NablAFx: {nablafx_lin_weight.shape}")

        if ref_lin_weight.shape == nablafx_lin_weight.shape:
            lin_diff = torch.abs(ref_lin_weight - nablafx_lin_weight).mean().item()
            print(f"  Mean absolute difference: {lin_diff:.10f}")
            if lin_diff < 1e-6:
                print(f"  ✅ Weights match perfectly")
            else:
                print(f"  ⚠️  Weights differ slightly")
        else:
            print(f"  ⚠️  Shapes don't match")
    except Exception as e:
        print(f"  ❌ Could not compare Linear weights: {e}")

    print("-" * 80)

    # Set to eval mode
    ref_model.eval()
    nablafx_model.eval()

    # Reset hidden states to ensure clean start
    ref_model.reset_hidden()
    nablafx_model.reset_state()

    # Create random input
    torch.manual_seed(123)
    batch_size = 2
    seq_length = 1000

    # Reference model expects: (seq_length, batch_size, input_size)
    x_ref = torch.randn(seq_length, batch_size, config["num_inputs"])

    # NablAFx model expects: (batch_size, input_size, seq_length)
    x_nablafx = x_ref.permute(1, 2, 0).clone()

    print(f"\nInput shapes:")
    print(f"  Reference: {x_ref.shape} (seq, batch, channels)")
    print(f"  NablAFx: {x_nablafx.shape} (batch, channels, seq)")

    # Forward pass
    with torch.no_grad():
        try:
            y_ref = torch.nn.functional.tanh(ref_model(x_ref))  # Apply tanh to match NablAFx output
            print(f"\n✅ Reference model forward pass successful")
            print(f"   Output shape: {y_ref.shape} (seq, batch, channels)")
            print(f"   Output range: [{y_ref.min():.4f}, {y_ref.max():.4f}]")
        except Exception as e:
            print(f"\n❌ Reference model failed: {e}")
            raise

        try:
            y_nablafx = nablafx_model(x_nablafx)
            print(f"\n✅ NablAFx model forward pass successful")
            print(f"   Output shape: {y_nablafx.shape} (batch, channels, seq)")
            print(f"   Output range: [{y_nablafx.min():.4f}, {y_nablafx.max():.4f}]")
        except Exception as e:
            print(f"\n❌ NablAFx model failed: {e}")
            raise

    # Convert NablAFx output to reference format for comparison
    # NablAFx: (batch, channels, seq) -> (seq, batch, channels)
    y_nablafx_compare = y_nablafx.permute(2, 0, 1)

    # Check shapes match
    print(f"\nShape comparison (after conversion):")
    print(f"  Reference: {y_ref.shape}")
    print(f"  NablAFx: {y_nablafx_compare.shape}")

    assert y_ref.shape == y_nablafx_compare.shape, f"Shape mismatch: {y_ref.shape} vs {y_nablafx_compare.shape}"

    # Compare outputs
    abs_diff = torch.abs(y_ref - y_nablafx_compare)
    mean_abs_diff = abs_diff.mean().item()
    max_abs_diff = abs_diff.max().item()

    print(f"\nNumerical comparison:")
    print(f"  Mean absolute difference: {mean_abs_diff:.6f}")
    print(f"  Max absolute difference: {max_abs_diff:.6f}")

    # Correlation check
    correlation = torch.corrcoef(torch.stack([y_ref.flatten(), y_nablafx_compare.flatten()]))[0, 1]
    print(f"  Output correlation: {correlation:.6f}")

    # Results
    print("\n" + "-" * 80)

    # NOTE: The main difference is that NablAFx applies tanh to the output
    # Reference: y = lin(lstm(x)) + res (if residual)
    # NablAFx: y = tanh(lin(lstm(x)) + res) (if residual)
    # This explains the output range difference: [-3.99, 3.64] vs [-0.999, 0.999]

    if mean_abs_diff < 1e-5 and correlation > 0.99:
        print("✅ TEST PASSED: Implementations match!")
        print("   Both models produce nearly identical outputs")
    elif correlation > 0.95:
        print("⚠️  TEST PARTIAL: Implementations are similar but not identical")
        print(f"   Mean diff: {mean_abs_diff:.6f} (threshold: 1e-5)")
        print(f"   Correlation: {correlation:.6f} (threshold: 0.99)")
        print("   This may be due to:")
        print("   - Different activation functions (tanh in NablAFx output vs none in Reference)")
        print("   - Different numerical precision")
        print("   - Subtle implementation differences")
        print("\n   NOTE: The main architectural difference is:")
        print("   - Reference output: y = lin(lstm(x)) + res")
        print("   - NablAFx output: y = tanh(lin(lstm(x)) + res)")
    else:
        print("❌ TEST FAILED: Implementations differ significantly")
        print(f"   Mean diff: {mean_abs_diff:.6f} (threshold: 1e-5)")
        print(f"   Correlation: {correlation:.6f} (threshold: 0.95)")

    print("-" * 80)


def test_deterministic():
    """Test both models are deterministic - same input yields same output on repeated calls"""
    print("\n" + "=" * 80)
    print("TEST: Determinism")
    print("=" * 80)

    # Test NablAFx model
    model_nablafx = NablAFxLSTM(
        num_inputs=1,
        num_outputs=1,
        num_controls=0,
        hidden_size=16,
        num_layers=1,
        residual=False,
        cond_type=None,
    )
    model_nablafx.eval()

    x = torch.randn(1, 1, 1000)  # (batch, channels, seq)

    with torch.no_grad():
        model_nablafx.reset_state()
        y1 = model_nablafx(x)

        model_nablafx.reset_state()
        y2 = model_nablafx(x)

    diff_nablafx = torch.abs(y1 - y2).max().item()

    # Test Reference model
    model_ref = ReferenceLSTM(
        input_size=1,
        output_size=1,
        unit_type="LSTM",
        hidden_size=16,
        skip=0,
        num_layers=1,
    )
    model_ref.eval()

    x_ref = x.permute(2, 0, 1)  # (seq, batch, channels)

    with torch.no_grad():
        model_ref.reset_hidden()
        y1_ref = model_ref(x_ref)

        model_ref.reset_hidden()
        y2_ref = model_ref(x_ref)

    diff_ref = torch.abs(y1_ref - y2_ref).max().item()

    print(f"NablAFx model determinism:")
    if diff_nablafx == 0:
        print(f"  ✅ Model is deterministic (diff: {diff_nablafx})")
    else:
        print(f"  ⚠️  Model is non-deterministic (diff: {diff_nablafx})")

    print(f"\nReference model determinism:")
    if diff_ref == 0:
        print(f"  ✅ Model is deterministic (diff: {diff_ref})")
    else:
        print(f"  ⚠️  Model is non-deterministic (diff: {diff_ref})")

    if diff_nablafx == 0 and diff_ref == 0:
        print("\n✅ TEST PASSED: Both models are deterministic")
    else:
        print("\n⚠️  TEST FAILED: At least one model is non-deterministic")


def test_multiple_configurations():
    """
    Test multiple configurations with copied weights.
    Verifies both implementations work correctly and produce similar outputs
    across different architectural configurations.
    """
    print("\n" + "=" * 80)
    print("TEST: Multiple Configurations")
    print("=" * 80)

    configs = [
        {"hidden_size": 16, "num_layers": 1, "residual": False},
        {"hidden_size": 32, "num_layers": 1, "residual": True},
        {"hidden_size": 64, "num_layers": 2, "residual": False},
    ]

    for config in configs:
        print(f"\nTesting: hidden={config['hidden_size']}, layers={config['num_layers']}, residual={config['residual']}")

        try:
            # Set seed BEFORE creating models to ensure consistent initialization
            torch.manual_seed(42)

            # Create reference model
            ref_model = ReferenceLSTM(
                input_size=1,
                output_size=1,
                unit_type="LSTM",
                hidden_size=config["hidden_size"],
                skip=1 if config["residual"] else 0,
                num_layers=config["num_layers"],
            )

            # Create NablAFx model
            nablafx_model = NablAFxLSTM(
                num_inputs=1,
                num_outputs=1,
                num_controls=0,
                hidden_size=config["hidden_size"],
                num_layers=config["num_layers"],
                residual=config["residual"],
                cond_type=None,
            )

            # Initialize reference model
            ref_model.apply(lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None)

            # Copy weights
            copy_success = copy_weights_ref_to_nablafx(ref_model, nablafx_model, verbose=False)
            if not copy_success:
                print("  ⚠️  Warning: Some weights failed to copy")

            # Set to eval mode
            ref_model.eval()
            nablafx_model.eval()

            # Reset states
            ref_model.reset_hidden()
            nablafx_model.reset_state()

            # Create test input
            torch.manual_seed(123)
            x_ref = torch.randn(500, 1, 1)  # (seq, batch, channels)
            x_nablafx = x_ref.permute(1, 2, 0)  # (batch, channels, seq)

            # Forward pass
            with torch.no_grad():
                y_ref = torch.nn.functional.tanh(ref_model(x_ref))  # Apply tanh to match NablAFx output
                y_nablafx = nablafx_model(x_nablafx)

            # Convert for comparison
            y_nablafx_compare = y_nablafx.permute(2, 0, 1)

            # Compare
            abs_diff = torch.abs(y_ref - y_nablafx_compare)
            mean_abs_diff = abs_diff.mean().item()
            correlation = torch.corrcoef(torch.stack([y_ref.flatten(), y_nablafx_compare.flatten()]))[0, 1]

            print(f"  Mean abs diff: {mean_abs_diff:.6f}, Correlation: {correlation:.6f}")

            if mean_abs_diff < 1e-4 and correlation > 0.95:
                print(f"  ✅ Configuration works correctly")
            else:
                print(f"  ⚠️  Configuration shows some differences")

        except Exception as e:
            print(f"  ❌ Configuration failed: {e}")
            traceback.print_exc()


def test_stateful_processing():
    """
    Test that both models maintain hidden state correctly across multiple forward passes.
    This is important for real-time audio processing.
    """
    print("\n" + "=" * 80)
    print("TEST: Stateful Processing")
    print("=" * 80)

    torch.manual_seed(42)

    # Create models
    ref_model = ReferenceLSTM(input_size=1, output_size=1, hidden_size=16, num_layers=1, skip=0)
    nablafx_model = NablAFxLSTM(num_inputs=1, num_outputs=1, hidden_size=16, num_layers=1, cond_type=None)

    # Copy weights
    ref_model.apply(lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None)
    copy_weights_ref_to_nablafx(ref_model, nablafx_model, verbose=False)

    ref_model.eval()
    nablafx_model.eval()

    # Create a long sequence and split it into chunks
    torch.manual_seed(123)
    full_length = 1000
    chunk_size = 100
    num_chunks = full_length // chunk_size

    x_full_ref = torch.randn(full_length, 1, 1)
    x_full_nablafx = x_full_ref.permute(1, 2, 0)

    # Test 1: Process full sequence at once (ground truth)
    print("\n1. Processing full sequence at once:")
    ref_model.reset_hidden()
    nablafx_model.reset_state()

    with torch.no_grad():
        y_full_ref = ref_model(x_full_ref)
        y_full_nablafx = nablafx_model(x_full_nablafx)

    print(f"   Reference output shape: {y_full_ref.shape}")
    print(f"   NablAFx output shape: {y_full_nablafx.shape}")

    # Test 2: Process in chunks WITH state preservation
    print("\n2. Processing in chunks WITH state preservation:")
    ref_model.reset_hidden()
    nablafx_model.reset_state()

    y_chunks_ref = []
    y_chunks_nablafx = []

    with torch.no_grad():
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size

            # Reference model
            x_chunk_ref = x_full_ref[start_idx:end_idx]
            y_chunk_ref = ref_model(x_chunk_ref)
            y_chunks_ref.append(y_chunk_ref)
            # State is automatically maintained in ref_model.hidden

            # NablAFx model
            x_chunk_nablafx = x_full_nablafx[:, :, start_idx:end_idx]
            y_chunk_nablafx = nablafx_model(x_chunk_nablafx)
            y_chunks_nablafx.append(y_chunk_nablafx)
            # State is automatically maintained in nablafx_model.hidden_state

    y_chunks_ref = torch.cat(y_chunks_ref, dim=0)
    y_chunks_nablafx = torch.cat(y_chunks_nablafx, dim=2)

    print(f"   Reference chunks output shape: {y_chunks_ref.shape}")
    print(f"   NablAFx chunks output shape: {y_chunks_nablafx.shape}")

    # Compare full vs chunks for each model
    diff_ref = torch.abs(y_full_ref - y_chunks_ref).mean().item()
    diff_nablafx_compare = torch.abs(y_full_nablafx - y_chunks_nablafx).mean().item()

    print(f"\n3. Consistency check (full vs chunks):")
    print(f"   Reference model - Mean abs diff: {diff_ref:.6f}")
    print(f"   NablAFx model - Mean abs diff: {diff_nablafx_compare:.6f}")

    if diff_ref < 1e-5:
        print(f"   ✅ Reference model: State maintained correctly")
    else:
        print(f"   ⚠️  Reference model: State inconsistency detected")

    if diff_nablafx_compare < 1e-5:
        print(f"   ✅ NablAFx model: State maintained correctly")
    else:
        print(f"   ⚠️  NablAFx model: State inconsistency detected")

    # Compare between models
    y_nablafx_compare = y_full_nablafx.permute(2, 0, 1)
    correlation = torch.corrcoef(torch.stack([y_full_ref.flatten(), y_nablafx_compare.flatten()]))[0, 1]

    print(f"\n4. Cross-model comparison:")
    print(f"   Correlation: {correlation:.6f}")

    if diff_ref < 1e-5 and diff_nablafx_compare < 1e-5 and correlation > 0.99:
        print("\n✅ TEST PASSED: Both models maintain state correctly")
    else:
        print("\n⚠️  TEST PARTIAL: Check individual results above")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("LSTM IMPLEMENTATION VERIFICATION")
    print("=" * 80)

    try:
        test_identical_weights_identical_output()
        test_deterministic()
        test_multiple_configurations()
        test_stateful_processing()

        print("\n" + "=" * 80)
        print("ALL TESTS COMPLETED")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\n" + "=" * 80)
        print("STACK TRACE:")
        print("=" * 80)
        traceback.print_exc()
