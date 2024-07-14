"""
   Copyright (c) 2024, UChicago Argonne, LLC
   All Rights Reserved

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
import numpy as np
import os 
import struct
import logging
from . import memory_pool

from dlio_benchmark.common.constants import MODULE_DATA_READER
from dlio_benchmark.reader.reader_handler import FormatReader
from dlio_benchmark.utils.utility import Profile

dlp = Profile(MODULE_DATA_READER)


class NPYReader(FormatReader):
    """
    Reader for NPY files
    """

    @dlp.log_init
    def __init__(self, dataset_type, thread_index, epoch, mempool=None):
        super().__init__(dataset_type, thread_index)
        self.mempool = mempool
        self.allocations = dict()

    @dlp.log
    def open(self, filename):        
        super().open(filename)

        file_size = os.path.getsize(filename)
        buffer_size = ((file_size + 4096 - 1) // 4096) * 4096
        buf = None
        try:
            buf = self.mempool.alloc(buffer_size)
            self.allocations[filename] = buf
            data = self.read_file(filename, file_size, buffer_size, buf)            
            return data            
            
        # If we fail to allocate memory, we will fall back to the default numpy load. 
        # This will ensure optimal performance when the size is within the expected size, but will still function when it is not.
        except memory_pool.AllocationFailed as e:
            return np.load(filename)
        
        except Exception as e:
            if buf != None:
                self.mempool.free(buf)
                self.allocations.pop(filename)
            raise e
    
    def read_file(self, filename, file_size, buffer_size, buf):
        try:
            # Open the file with O_DIRECT
            fd = os.open(filename, os.O_RDONLY | os.O_DIRECT)
            mem_view = memoryview(buf)
            # Read the file into the buffer
            bytes_read = os.readv(fd, [mem_view[0:buffer_size]])
            if bytes_read != file_size:
                raise IOError(f"Could not read the entire file. Expected {file_size} bytes, got {bytes_read} bytes")

            return self.parse_npy(buf, mem_view)
        finally:
            os.close(fd)

    def parse_npy(self, buf, mem_view):
        # Verify the magic string
        if buf[:6] != b'\x93NUMPY':
            raise ValueError("This is not a valid .npy file.")

        # Read version information
        major, minor = struct.unpack('<BB', buf[6:8])
        if major == 1:
            header_len = struct.unpack('<H', buf[8:10])[0]
            header = buf[10:10 + header_len]
        elif major == 2:
            header_len = struct.unpack('<I', buf[8:12])[0]
            header = buf[12:12 + header_len]
        else:
            raise ValueError(f"Unsupported .npy file version: {major}.{minor}")

        # Parse the header
        header_dict = eval(header.decode('latin1'))
        dtype = np.dtype(header_dict['descr'])
        shape = header_dict['shape']
        fortran_order = header_dict['fortran_order']

        # Calculate the data offset
        data_offset = (10 + header_len) if major == 1 else (12 + header_len)
        data_size = np.prod(shape) * dtype.itemsize

        # Load the array data
        data = np.ndarray(shape, dtype=dtype, buffer=mem_view[data_offset:data_offset + data_size])

        # If the array is in Fortran order, convert it
        if fortran_order:
            data = np.asfortranarray(data)

        return data

    @dlp.log
    def close(self, filename):
        super().close(filename)

        # This may happen if we go through the "slow path" (e.g np.load)
        if filename not in self.allocations:
            return
        buf = self.allocations[filename]
        self.mempool.free(buf)
        del(self.allocations[filename])

    @dlp.log
    def get_sample(self, filename, sample_index):
        super().get_sample(filename, sample_index)
        image = self.open_file_map[filename][..., sample_index]
        dlp.update(image_size=image.nbytes)

    def next(self):
        for batch in super().next():
            yield batch

    @dlp.log
    def read_index(self, image_idx, step):
        return super().read_index(image_idx, step)

    @dlp.log
    def finalize(self):
        return super().finalize()

    def is_index_based(self):
        return True

    def is_iterator_based(self):
        return True