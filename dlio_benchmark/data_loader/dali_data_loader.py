"""
   Copyright (c) 2022, UChicago Argonne, LLC
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
from time import time
import logging
import math
import numpy as np
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali as dali
from nvidia.dali.plugin.pytorch import DALIGenericIterator

from dlio_benchmark.common.constants import MODULE_DATA_LOADER
from dlio_benchmark.common.enumerations import Shuffle, DataLoaderType, DatasetType
from dlio_benchmark.data_loader.base_data_loader import BaseDataLoader
from dlio_benchmark.reader.reader_factory import ReaderFactory
from dlio_benchmark.utils.utility import utcnow, get_rank, timeit, Profile

dlp = Profile(MODULE_DATA_LOADER)


class DaliDataset(object):

    def __init__(self, format_type, dataset_type, epoch, num_samples, batch_size, thread_index):
        self.format_type = format_type
        self.dataset_type = dataset_type
        self.epoch = epoch
        self.num_samples = num_samples
        self.num_images_read = 0
        self.batch_size = batch_size
        self.reader = ReaderFactory.get_reader(type=self.format_type,
                                               dataset_type=self.dataset_type,
                                               thread_index=thread_index,
                                               epoch_number=self.epoch)
        self.item = self.reader.next()
        self.is_last = 0

    def __call__(self, sample_info):
        self.num_images_read += 1
        step = int(math.ceil(self.num_images_read / self.batch_size))
        sample_idx = sample_info.idx_in_epoch
        logging.debug(f"{utcnow()} Reading {sample_idx} {sample_info.iteration} {self.num_samples} {self.batch_size}")
        if sample_info.iteration >= self.num_samples // self.batch_size:
            # Indicate end of the epoch
            raise StopIteration()
        with Profile(MODULE_DATA_LOADER, epoch=self.epoch,image_idx=sample_idx, step=step):
            image = self.reader.read_index(sample_idx, step)
        return image, np.uint8([sample_idx])


class DaliDataLoader(BaseDataLoader):
    @dlp.log_init
    def __init__(self, format_type, dataset_type, epoch):
        super().__init__(format_type, dataset_type, epoch, DataLoaderType.DALI)
        self.pipeline = None

    @dlp.log
    def read(self):
        parallel = True if self._args.read_threads > 0 else False
        self.pipelines = []
        num_threads = 1
        if self._args.read_threads > 0:
            num_threads = self._args.read_threads
        prefetch_size = 2
        if self._args.prefetch_size > 0:
            prefetch_size = self._args.prefetch_size
        # None executes pipeline on CPU and the reader does the batching
        dataset = DaliDataset(self.format_type, self.dataset_type, self.epoch_number, self.num_samples, self.batch_size, 0)
        self.pipeline = Pipeline(batch_size=self.batch_size, num_threads=num_threads, device_id=None, py_num_workers=num_threads,
                            prefetch_queue_depth=prefetch_size, py_start_method='fork', exec_async=True)
        with self.pipeline:
            images, labels = fn.external_source(source=dataset, num_outputs=2, dtype=[types.UINT8, types.UINT8],
                                                parallel=True, batch=False)
            self.pipeline.set_outputs(images, labels)

        self.pipeline.build()
        logging.debug(f"{utcnow()} Starting {num_threads} pipelines by {self._args.my_rank} rank ")


    @dlp.log
    def next(self):
        super().next()
        #DALIGenericIterator(self.pipelines, ['data', 'label'])
        logging.debug(f"{utcnow()} Iterating pipelines by {self._args.my_rank} rank ")
        for step in range(self.num_samples // self.batch_size):
            outputs = self.pipeline.run()
            logging.debug(f"{utcnow()} Output batch {step} {len(outputs)}")
            for batch in outputs:
                yield outputs



    @dlp.log
    def finalize(self):
        pass
