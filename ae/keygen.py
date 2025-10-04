import os
from typing import Iterator, Optional
import warnings

CURVE_ORDER = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141

class RandomKeyGenerator:
    """
    Random key generator with mandatory GPU acceleration via PyTorch.
    This class will fail to initialize if a CUDA-enabled GPU is not available.
    """
    def __init__(self, use_gpu: bool = True, torch_device: Optional[str] = None, force_gpu: bool = True):
        if not use_gpu:
            raise ValueError("use_gpu must be True. This project requires GPU acceleration.")
        
        self.force_gpu = force_gpu
        self._torch = None
        self.torch_device = None

        try:
            import torch
            self._torch = torch
        except ImportError:
            raise RuntimeError(
                "PyTorch is not installed. Please install it with CUDA support: "
                "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
            )

        # MODIFIED: Strict GPU check
        if not self._torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. This project is configured to run exclusively on a supported NVIDIA GPU. "
                "Please check your CUDA installation and GPU drivers."
            )

        if torch_device is None:
            self.torch_device = 'cuda'
        else:
            if 'cuda' not in torch_device:
                raise ValueError(f"Invalid device '{torch_device}'. Only CUDA devices are supported.")
            self.torch_device = torch_device

        # Set the default device and perform a test operation
        try:
            self._torch.cuda.set_device(0)
            test_tensor = self._torch.tensor([1.0], device=self.torch_device)
            print(f"[GPU] Successfully initialized on {self._torch.get_device_name(0)}")
        except Exception as e:
            raise RuntimeError(f"Failed to perform a test operation on GPU: {e}")

    def random_keys(self, batch_size: int = 1) -> Iterator[bytes]:
        """Generates random keys exclusively on the GPU."""
        yield from self._gpu_stream(batch_size)

    def _cpu_stream(self, batch_size: int) -> Iterator[bytes]:
        # MODIFIED: This method now raises an error to prevent fallback
        raise RuntimeError("CPU fallback is disabled. The process requires a GPU to continue.")

    def _gpu_stream(self, batch_size: int) -> Iterator[bytes]:
        """
        Generates random keys on the GPU using vectorized operations.
        This method is optimized for high throughput on CUDA devices.
        """
        torch = self._torch
        device = self.torch_device
        order = CURVE_ORDER

        # Increase batch size for better GPU utilization
        effective_batch_size = batch_size * 4
        
        while True:
            with torch.cuda.device(device):
                # Generate a large batch of random numbers on the GPU
                # Using uint64 for intermediate steps to handle large numbers
                random_uint64 = torch.randint(
                    low=1, 
                    high=order, 
                    size=(effective_batch_size,), 
                    dtype=torch.uint64, 
                    device=device
                )

                # Convert uint64 to 32-byte big-endian format
                # This is a vectorized approach to avoid Python loops on the GPU
                # Reshape to view each number as 8 bytes
                byte_view = random_uint64.view(torch.uint8)
                
                # We need 32 bytes, so we'll create a tensor of zeros and fill it
                # This is more efficient than complex bit-shifting on GPU for this case
                full_bytes = torch.zeros((effective_batch_size, 32), dtype=torch.uint8, device=device)
                
                # Copy the 8 bytes from uint64 to the last 8 bytes of the 32-byte array
                # This represents the number in big-endian format
                full_bytes[:, 24:] = byte_view

                # Convert to CPU and then to Python bytes
                # This is the bottleneck, but necessary to get Python objects
                cpu_bytes = full_bytes.cpu().numpy().tobytes()
                
                # Yield the keys one by one
                for i in range(effective_batch_size):
                    yield cpu_bytes[i*32 : (i+1)*32]


class IncrementalKeyGenerator:
    """Generate keys in a specific numeric range (CPU-bound)."""
    def __init__(self, start: int, end: int):
        print("[WARNING] IncrementalKeyGenerator is CPU-bound.")
        self.start = start
        self.end = end
        self.current = start

    def keys_in_range(self):
        """Generator function that yields keys in the specified range."""
        current = self.start
        while current <= self.end:
            key = current.to_bytes(32, 'big')
            current += 1
            yield key