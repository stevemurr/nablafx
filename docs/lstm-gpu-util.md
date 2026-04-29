in absolute terms because you trade many tiny launches for fewer larger ones.                                  
    3. Block size on the controller. The DynamicController emits params at block_size=128. Larger block size (e.g. 
    256, 512) means fewer LSTM steps per audio second — directly reduces the sequential bottleneck. Tradeoff:      
    coarser parameter modulation, which for an EQ preset (slowly-varying) is fine.                                 
    4. torch.compile — we have it disabled (compile=false) because of an earlier inductor hang. If that's          
    revisited, compile fuses the per-timestep python overhead into a single CUDA graph and typically 2-3x's
    tiny-RNN throughput. Highest-risk option, biggest payoff.
    5. CUDA-fused LSTM kernel. PyTorch's default LSTM already uses cuDNN's fused kernel for standard configs, but
    only if batch_first=True and the sequence dim is reasonable. Worth checking that no custom unrolling is being
    used in nablafx/processors/ddsp.py controller code.
    6. Channels_last + cudnn.benchmark=true — benchmark: true is already set in gb.yaml. Good.
    7. AMP (bf16-mixed) — we ruled this out earlier for the FFT path (ComplexHalf NaN'd in fft_freqz). But the LSTM
     forward and most non-FFT ops would benefit. A targeted with autocast() around just the controller would buy
    speed without re-introducing the NaN — though it's surgical work.
    8. Dataloader. Worth checking num_workers and prefetch_factor aren't starving the GPU. The fanout config has
    preload=true for trainval which puts the dataset in RAM, so dataloader cost should be near zero — but worth
    confirming with nvidia-smi dmon (look for GPU util sitting at 100% vs flickering).

    Best single move for our setup: raise batch_size to 64 (or 128 if it fits) and dyn_num_layers/dyn_hidden_dim if
     there's a knob. Both attack throughput and the per-batch noise floor at the same time.
