"""
Unified FPN: High-Performance CUDA Feature Pyramid Network Implementation

A heavily optimized CUDA implementation of Feature Pyramid Networks that fuses
multiple operations into unified kernels for maximum performance.

Key Features:
- Fused lateral convolution + upsampling + element-wise addition
- Memory-optimized shared memory usage  
- Multi-stream parallel execution
- Support for FP32/FP16 mixed precision
- PyTorch integration with autograd support
- Comprehensive benchmarking and profiling

Example Usage:
    import torch
    from unified_fpn import UnifiedFPN
    
    # Create FPN with 256 output channels
    fpn = UnifiedFPN(output_channels=256)
    
    # Backbone features C2-C6 from ResNet
    backbone_features = [
        torch.randn(1, 256, 200, 304, device='cuda'),   # C2
        torch.randn(1, 512, 100, 152, device='cuda'),   # C3  
        torch.randn(1, 1024, 50, 76, device='cuda'),    # C4
        torch.randn(1, 2048, 25, 38, device='cuda'),    # C5
        torch.randn(1, 2048, 13, 19, device='cuda')     # C6
    ]
    
    # Forward pass - returns P2-P5 features
    pyramid_features = fpn(backbone_features)
    
    # Benchmark performance
    avg_time, bandwidth, compute_util = fpn.benchmark(backbone_features)
    print(f"Average time: {avg_time:.2f}ms")
    print(f"Memory bandwidth: {bandwidth:.1f} GB/s") 
    print(f"Compute utilization: {compute_util:.1f}%")
"""

__version__ = "1.0.0"
__author__ = "Advanced CUDA Developer"
__email__ = "dev@example.com"

# Import core components
try:
    from .unified_fpn import UnifiedFPN
    from .unified_fpn import fpn_lateral_conv, fpn_upsample_bilinear, fpn_forward_pytorch
    
    # Export main classes and functions
    __all__ = [
        'UnifiedFPN',
        'fpn_lateral_conv', 
        'fpn_upsample_bilinear',
        'fpn_forward_pytorch',
        'benchmark_fpn',
        'profile_fpn',
        'get_device_info'
    ]
    
except ImportError as e:
    import warnings
    warnings.warn(f"Failed to import CUDA extensions: {e}. "
                 f"Make sure you have CUDA installed and the package was compiled correctly.")
    
    # Provide CPU fallback implementations
    class UnifiedFPN:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("CUDA extensions not available. CPU fallback not implemented.")
    
    def fpn_lateral_conv(*args, **kwargs):
        raise RuntimeError("CUDA extensions not available.")
    
    def fpn_upsample_bilinear(*args, **kwargs): 
        raise RuntimeError("CUDA extensions not available.")
    
    def fpn_forward_pytorch(*args, **kwargs):
        raise RuntimeError("CUDA extensions not available.")

import torch

def benchmark_fpn(backbone_features, num_iterations=100, warmup_iterations=10):
    """
    Comprehensive benchmark of FPN performance.
    
    Args:
        backbone_features: List of 5 tensors (C2-C6) on CUDA device
        num_iterations: Number of timing iterations
        warmup_iterations: Number of warmup iterations
        
    Returns:
        dict: Benchmark results including timing, bandwidth, memory usage
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available for benchmarking")
    
    fpn = UnifiedFPN()
    
    # Warmup
    for _ in range(warmup_iterations):
        _ = fpn(backbone_features)
    
    torch.cuda.synchronize()
    
    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(num_iterations):
        pyramid_features = fpn(backbone_features)
    end_event.record()
    
    torch.cuda.synchronize()
    
    total_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = total_time_ms / num_iterations
    
    # Get detailed metrics
    detailed_metrics = fpn.benchmark(backbone_features, num_iterations)
    
    # Calculate memory usage
    total_memory_mb = sum(t.numel() * t.element_size() for t in backbone_features) / (1024 * 1024)
    total_memory_mb += sum(t.numel() * t.element_size() for t in pyramid_features) / (1024 * 1024)
    
    return {
        'avg_time_ms': avg_time_ms,
        'throughput_fps': 1000.0 / avg_time_ms,
        'memory_bandwidth_gb_s': detailed_metrics[1],
        'compute_utilization': detailed_metrics[2],
        'memory_usage_mb': total_memory_mb,
        'detailed_metrics': detailed_metrics
    }

def profile_fpn(backbone_features, output_file="fpn_profile.json"):
    """
    Profile FPN execution and save results.
    
    Args:
        backbone_features: List of 5 tensors (C2-C6) on CUDA device
        output_file: Path to save profiling results
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available for profiling")
    
    fpn = UnifiedFPN()
    
    # Enable profiling
    torch.cuda.profiler.start()
    
    # Run FPN
    pyramid_features = fpn(backbone_features)
    
    # Stop profiling  
    torch.cuda.profiler.stop()
    
    # Get performance breakdown
    timing_breakdown, metrics = fpn.get_performance_breakdown()
    
    profile_data = {
        'kernel_times_ms': timing_breakdown.cpu().numpy().tolist(),
        'total_time_ms': metrics[0].item(),
        'memory_bandwidth_gb_s': metrics[1].item(),
        'compute_utilization': metrics[2].item(),
        'peak_memory_mb': metrics[3].item(),
        'num_kernel_launches': int(metrics[4].item())
    }
    
    import json
    with open(output_file, 'w') as f:
        json.dump(profile_data, f, indent=2)
    
    print(f"Profiling results saved to {output_file}")
    return profile_data

def get_device_info():
    """
    Get detailed CUDA device information.
    
    Returns:
        dict: Device properties and capabilities
    """
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    
    return {
        'device_name': props.name,
        'compute_capability': f"{props.major}.{props.minor}",
        'total_memory_gb': props.total_memory / (1024**3),
        'multiprocessor_count': props.multi_processor_count,
        'max_threads_per_block': props.max_threads_per_block,
        'max_threads_per_multiprocessor': props.max_threads_per_multiprocessor,
        'memory_clock_rate_mhz': props.memory_clock_rate / 1000,
        'memory_bus_width_bits': props.memory_bus_width,
        'l2_cache_size_mb': props.l2_cache_size / (1024 * 1024),
        'theoretical_bandwidth_gb_s': 2 * props.memory_clock_rate * props.memory_bus_width / 8 / 1e6
    }

def check_installation():
    """
    Check if the installation is working correctly.
    
    Returns:
        bool: True if installation is working
    """
    try:
        # Check CUDA availability
        if not torch.cuda.is_available():
            print("❌ CUDA not available")
            return False
        
        print("✅ CUDA available")
        
        # Check device info
        device_info = get_device_info()
        print(f"✅ GPU: {device_info['device_name']}")
        print(f"✅ Compute capability: {device_info['compute_capability']}")
        
        # Test basic functionality
        fpn = UnifiedFPN(output_channels=64)
        print("✅ UnifiedFPN created successfully")
        
        # Test with small tensors
        test_features = [
            torch.randn(1, 64, 8, 8, device='cuda'),
            torch.randn(1, 128, 4, 4, device='cuda'), 
            torch.randn(1, 256, 2, 2, device='cuda'),
            torch.randn(1, 512, 1, 1, device='cuda'),
            torch.randn(1, 512, 1, 1, device='cuda')
        ]
        
        pyramid_features = fpn(test_features)
        print("✅ Forward pass successful")
        
        # Verify output shapes
        expected_shapes = [
            (1, 64, 8, 8),   # P2
            (1, 64, 4, 4),   # P3
            (1, 64, 2, 2),   # P4
            (1, 64, 1, 1)    # P5
        ]
        
        for i, (output, expected) in enumerate(zip(pyramid_features, expected_shapes)):
            if output.shape != expected:
                print(f"❌ Wrong output shape for P{i+2}: got {output.shape}, expected {expected}")
                return False
        
        print("✅ Output shapes correct")
        print("✅ Installation check passed!")
        return True
        
    except Exception as e:
        print(f"❌ Installation check failed: {e}")
        return False

# Version check
def check_cuda_version():
    """Check CUDA version compatibility."""
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
        print(f"PyTorch CUDA version: {cuda_version}")
        
        # Check if version is supported
        supported_versions = ['11.7', '11.8', '12.0', '12.1', '12.2']
        if any(cuda_version.startswith(v) for v in supported_versions):
            print("✅ CUDA version supported")
        else:
            print(f"⚠️  CUDA version {cuda_version} may not be fully supported")
    else:
        print("❌ CUDA not available")

# Auto-run installation check on import (can be disabled)
import os
if os.environ.get('UNIFIED_FPN_SKIP_CHECK', '0') != '1':
    if torch.cuda.is_available():
        print("Unified FPN: Checking installation...")
        check_cuda_version()