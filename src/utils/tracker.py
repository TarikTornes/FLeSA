import tracemalloc
import psutil
import os
import gc
import tracemalloc


def track_memory():
    """
    Function to track memory usage
    """
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024  # in MB
    return mem

class MemoryProfiler:
    def __init__(self):
        self.initial_memory = track_memory()
        self.memories = []

    def log_memory(self, label=''):
        current_memory = track_memory()
        memory_diff = current_memory - self.initial_memory
        self.memories.append((label, current_memory, memory_diff))
        print(f"{label} - Memory Used: {current_memory:.2f} MB (Increase: {memory_diff:.2f} MB)")
        return current_memory


def trace_mem(line_info):
    current, peak = tracemalloc.get_traced_memory()
    print(f"[{line_info}] Current memory usage: {current / 1024:.2f} KB; Peak: {peak / 1024:.2f} KB")
