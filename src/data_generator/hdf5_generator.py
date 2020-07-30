import h5py
from numpy import random
import math

from src.data_generator.data_generator import DataGenerator
from src.utils.utility import progress
from shutil import copyfile


class HDF5Generator(DataGenerator):
    def __init__(self):
        super().__init__()
        self.chunk_size = self._arg_parser.args.chunk_size
        self.enable_chunking = self._arg_parser.args.enable_chunking

    def generate(self):
        super().generate()
        records = random.random((self._dimension, self._dimension, self.num_samples))
        record_labels = [0] * self.num_samples
        prev_out_spec = ""
        count = 0
        for i in range(0, int(self.num_files)):
            if i % self.comm_size == self.my_rank:
                progress(i+1, self.num_files, "Generating HDF5 Data")
                out_path_spec = "{}_{}_of_{}.h5".format(self._file_prefix, i+1, self.num_files)
                if count == 0:
                    prev_out_spec = out_path_spec
                    hf = h5py.File(out_path_spec, 'w')
                    if self.enable_chunking:
                        chunk_dimension = int(math.ceil(math.sqrt(self.chunk_size)))
                        if chunk_dimension > self._dimension:
                            chunk_dimension = self._dimension
                        hf.create_dataset('records', data=records, chunks=(chunk_dimension, chunk_dimension, 1))
                        hf.create_dataset('labels', data=record_labels)
                    else:
                        hf.create_dataset('records', data=records)
                        hf.create_dataset('labels', data=record_labels)
                    hf.close()
                    count += 1
                else:
                    copyfile(prev_out_spec, out_path_spec)
