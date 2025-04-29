import torch


class MemoryManager:
    def __init__(self, cleanup_frequency=100):
        self.cleanup_frequency = cleanup_frequency

    def cleanup(self, frame_count):
        if frame_count % self.cleanup_frequency == 0:
            if torch.cuda.is_available():
                # Clear CUDA cache
                torch.cuda.empty_cache()
                # Force garbage collection
                import gc
                gc.collect()

    @staticmethod
    def print_memory_stats():
        if torch.cuda.is_available():
            print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            print(f"GPU Memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
