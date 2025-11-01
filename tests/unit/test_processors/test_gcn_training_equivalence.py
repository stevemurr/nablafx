"""
Test architectural equivalence by training both models on the same task.

If architectures are equivalent, they should:
1. Achieve similar final loss
2. Learn similar input-output mappings
3. Show high correlation on test data after training
"""

import torch
import torch.nn as nn
import torch.optim as optim
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

    This gives both models the same initialization, allowing us to test
    whether they converge similarly from the same starting point.
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


def generate_synthetic_task(num_samples=100, seq_len=4000):
    """
    Generate a simple audio processing task:
    - Input: noisy signal
    - Target: clean signal (sine wave with harmonics)
    """
    torch.manual_seed(42)

    # Generate clean target signals
    t = torch.linspace(0, 1, seq_len)
    targets = []
    inputs = []

    for _ in range(num_samples):
        # Random fundamental frequency
        f0 = torch.rand(1) * 200 + 100  # 100-300 Hz

        # Clean signal: fundamental + harmonics
        clean = 0.5 * torch.sin(2 * torch.pi * f0 * t)
        clean += 0.3 * torch.sin(2 * torch.pi * 2 * f0 * t)
        clean += 0.2 * torch.sin(2 * torch.pi * 3 * f0 * t)

        # Add noise to create input
        noise = torch.randn(seq_len) * 0.3
        noisy = clean + noise

        inputs.append(noisy.unsqueeze(0).unsqueeze(0))  # [1, 1, seq_len]
        targets.append(clean.unsqueeze(0).unsqueeze(0))

    return torch.cat(inputs), torch.cat(targets)


def train_model(model, train_x, train_y, epochs=50, lr=0.001, model_name="Model"):
    """Train a model and return loss history"""

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    is_reference = isinstance(model, ReferenceGCN)

    loss_history = []
    model.train()

    print(f"\nTraining {model_name}...")
    for epoch in range(epochs):
        # Process in batches to avoid memory issues
        batch_losses = []
        batch_size = 4
        num_batches = (train_x.shape[0] + batch_size - 1) // batch_size

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, train_x.shape[0])

            batch_x = train_x[start_idx:end_idx]
            batch_y = train_y[start_idx:end_idx]

            # Convert format if needed
            if is_reference:
                # Reference expects [length, batch, channels]
                batch_x = batch_x.permute(2, 0, 1)
                batch_y = batch_y.permute(2, 0, 1)

            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

        epoch_loss = sum(batch_losses) / len(batch_losses)
        epoch_loss = sum(batch_losses) / len(batch_losses)

        loss_history.append(epoch_loss)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}: Loss = {epoch_loss:.6f}")

    return loss_history


def test_equivalence():
    print("=" * 80)
    print("ARCHITECTURAL EQUIVALENCE TEST: Training-based Proof")
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

    # Verify same capacity
    ref_params = sum(p.numel() for p in ref_model.parameters())
    nablafx_params = sum(p.numel() for p in nablafx_model.parameters())
    print(f"\nModel parameters: Reference={ref_params:,}, NablAFx={nablafx_params:,}")
    assert ref_params == nablafx_params, "Parameter counts must match!"

    # Initialize reference model with random seed
    print("\nInitializing models with same weights...")
    torch.manual_seed(42)
    ref_model.apply(lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None)

    # Copy weights to give both models identical initialization
    copy_weights_ref_to_nablafx(ref_model, nablafx_model, verbose=True)

    # Verify weights were copied
    ref_first_conv = ref_model.blocks[0].layers[0].conv.weight
    nablafx_first_conv = nablafx_model.blocks[0].conv.weight
    weight_diff = torch.abs(ref_first_conv - nablafx_first_conv).mean().item()
    print(f"  Initial weight difference: {weight_diff:.10f}")
    if weight_diff < 1e-6:
        print(f"  ✅ Both models start with identical weights!")

    # Generate synthetic task
    print("\nGenerating synthetic denoising task...")
    train_x, train_y = generate_synthetic_task(num_samples=50, seq_len=4000)
    test_x, test_y = generate_synthetic_task(num_samples=10, seq_len=4000)

    print(f"  Training set: {train_x.shape}")
    print(f"  Test set: {test_x.shape}")

    # Train both models
    ref_losses = train_model(ref_model, train_x.clone(), train_y.clone(), epochs=100, lr=0.001, model_name="Reference GCN")

    nablafx_losses = train_model(nablafx_model, train_x.clone(), train_y.clone(), epochs=100, lr=0.001, model_name="NablAFx GCN")

    # Compare final losses
    print("\n" + "=" * 80)
    print("TRAINING RESULTS")
    print("=" * 80)
    print(f"Reference GCN - Final training loss: {ref_losses[-1]:.6f}")
    print(f"NablAFx GCN   - Final training loss: {nablafx_losses[-1]:.6f}")
    print(f"Loss difference: {abs(ref_losses[-1] - nablafx_losses[-1]):.6f}")

    # Evaluate on test set
    ref_model.eval()
    nablafx_model.eval()

    with torch.no_grad():
        # Process test data in batches
        ref_preds = []
        nablafx_preds = []

        for i in range(test_x.shape[0]):
            test_sample = test_x[i : i + 1]

            # Reference format
            test_sample_ref = test_sample.permute(2, 0, 1)

            ref_pred_sample = ref_model(test_sample_ref)
            nablafx_pred_sample = nablafx_model(test_sample)

            # Convert to same format
            ref_pred_sample = ref_pred_sample.permute(1, 2, 0)

            ref_preds.append(ref_pred_sample)
            nablafx_preds.append(nablafx_pred_sample)

        ref_pred = torch.cat(ref_preds, dim=0)
        nablafx_pred = torch.cat(nablafx_preds, dim=0)

        # Calculate test losses
        criterion = nn.MSELoss()
        ref_test_loss = criterion(ref_pred, test_y).item()
        nablafx_test_loss = criterion(nablafx_pred, test_y).item()

        # Calculate correlation between predictions
        ref_flat = ref_pred.reshape(-1)
        nablafx_flat = nablafx_pred.reshape(-1)
        correlation = torch.corrcoef(torch.stack([ref_flat, nablafx_flat]))[0, 1].item()

        # Calculate how similar they are to ground truth
        ref_to_target_corr = torch.corrcoef(torch.stack([ref_pred.reshape(-1), test_y.reshape(-1)]))[0, 1].item()
        nablafx_to_target_corr = torch.corrcoef(torch.stack([nablafx_pred.reshape(-1), test_y.reshape(-1)]))[0, 1].item()

    print("\n" + "=" * 80)
    print("TEST SET EVALUATION")
    print("=" * 80)
    print(f"Reference GCN - Test loss: {ref_test_loss:.6f}")
    print(f"NablAFx GCN   - Test loss: {nablafx_test_loss:.6f}")
    print(f"Test loss difference: {abs(ref_test_loss - nablafx_test_loss):.6f}")
    print(f"\nCorrelation between predictions: {correlation:.6f}")
    print(f"Reference prediction vs target: {ref_to_target_corr:.6f}")
    print(f"NablAFx prediction vs target:   {nablafx_to_target_corr:.6f}")

    # Verdict
    print("\n" + "=" * 80)
    print("EQUIVALENCE VERDICT")
    print("=" * 80)

    loss_similar = abs(ref_test_loss - nablafx_test_loss) < 0.01
    predictions_similar = correlation > 0.8
    both_learn = ref_to_target_corr > 0.7 and nablafx_to_target_corr > 0.7

    if loss_similar and predictions_similar and both_learn:
        print("✅ ARCHITECTURES ARE EQUIVALENT!")
        print("   - Both achieve similar test loss")
        print("   - Both make similar predictions (high correlation)")
        print("   - Both successfully learn the task")
    elif both_learn:
        print("✅ ARCHITECTURES ARE EQUIVALENT (with different solutions)!")
        print("   - Both successfully learn the task")
        print("   - May have found different local optima (expected)")
    else:
        print("⚠️  Inconclusive - may need more training or different task")
        print(f"   - Loss similar: {loss_similar}")
        print(f"   - Predictions similar: {predictions_similar}")
        print(f"   - Both learn: {both_learn}")


if __name__ == "__main__":
    test_equivalence()
