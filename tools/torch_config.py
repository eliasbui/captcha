import torch
import os
def setup_torch_optimizations():
    """Setup optimizations based on available hardware"""
    
    # Always safe to set
    torch.backends.cudnn.benchmark = True
    
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name()}")
        
        # GPU-specific optimizations
        device_props = torch.cuda.get_device_properties(0)
        
        # TF32 - Only on Ampere+ GPUs (RTX 30/40 series, A100, etc.)
        if device_props.major >= 8:
            torch.set_float32_matmul_precision('medium')
            torch.backends.cudnn.allow_tf32 = True
            print("Optimization: TF32 enabled (Ampere+ GPU)")
        else:
            print(f"Optimization: TF32 not available on {device_props.name}")
        # Additional GPU optimizations
        torch.backends.cudnn.deterministic = False  # Faster but non-deterministic
        print("Optimization: Non-deterministic algorithms enabled")
        torch.backends.cudnn.enabled = True
        
        # Memory optimizations for GPU
        torch.cuda.empty_cache()  # Clear cache at startup
        
        # For newer PyTorch versions with CUDA graphs
        if hasattr(torch.cuda, 'is_current_stream_capturing'):
            print("Optimization: CUDA graphs support available")
            
    else:
        print("GPU not detected. Running Torch in CPU mode...")
        # CPU-specific optimizations
        torch.set_num_threads(os.cpu_count())
        print(f"CPU threads set to {os.cpu_count()} for Torch")