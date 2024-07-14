import ctypes 
import queue

class AllocationFailed(Exception):
    def __init__(self, message):
        super().__init__(message)

class AlignedMemoryPool():
    def __init__(self, alignment, pool_size, max_sample_size):
        self.alignment = alignment
        self.pool_size = pool_size
        self.max_sample_size = max_sample_size
        self.pool = queue.Queue()

        for i in range(pool_size):
            self.pool.put(self.allocate_aligned_buffer(self.max_sample_size+ alignment - 1))

    # This function allocates a buffer with the given size and passes its aligned offset to the caller
    def allocate_aligned_buffer(self, max_sample_size):
        buf_size = max_sample_size + (self.alignment - 1)
        raw_memory = bytearray(buf_size)
        ctypes_raw_type = (ctypes.c_char * buf_size)
        ctypes_raw_memory = ctypes_raw_type.from_buffer(raw_memory)
        raw_address = ctypes.addressof(ctypes_raw_memory)
        offset = raw_address % self.alignment
        offset_to_aligned = (self.alignment - offset) % self.alignment
        ctypes_aligned_type = (ctypes.c_char * (buf_size - offset_to_aligned))
        ctypes_aligned_memory = ctypes_aligned_type.from_buffer(raw_memory, offset_to_aligned)
        return ctypes_aligned_memory
    
    def alloc(self, size, block=True):
        if size > self.max_sample_size:
            raise AllocationFailed('Requested size is greater than the maximum sample size')
        
        return self.pool.get(block=block)        
    
    def free(self, buf):
        self.pool.put(buf)
   