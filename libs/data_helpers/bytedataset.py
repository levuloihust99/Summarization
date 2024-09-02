"""Define ByteDataset class.

ByteDataset is used to load big dataset and easily shuffling. A ByteDataset is specified by 
a `idxs.pkl` file and a `data.pkl` file.

The first 4 bytes of `idxs.pkl` is size of the dataset (number of samples),
i.e. maximum number of samples = 2^32 - 1 = 4,294,967,295
Each sample is stored in a variable-length number of bytes in `data.pkl`.
The position of sample i-th (bytes offset) in `data.pkl` is specified in `idxs.pkl`,
i.e. pos = 4 + i * idx_record_size,
where `idx_record_size` is number of bytes used to specify position of a sample, meaning 
that maximum size of `data.pkl` is about 2^48 bytes = 64 TiB.
"""

import os
import pickle
import logging
from typing import Text, Any

logger = logging.getLogger(__name__)


class ByteDataset:
    def __init__(
        self,
        data_path: Text,
        idx_record_size: int = 6,
        transform=None
    ):
        self.data_path = data_path
        self.idx_file = open(os.path.join(data_path, "idxs.pkl"), "rb+")
        self.data_file = open(os.path.join(data_path, "data.pkl"), "rb+")
        self.idx_record_size = idx_record_size
        self.transform = transform
    
    def __len__(self):
        self.idx_file.seek(0, 0)
        dataset_size = self.idx_file.read(4)
        dataset_size = int.from_bytes(dataset_size, byteorder='big', signed=False)
        return dataset_size
    
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start = idx.start
            stop = idx.stop or len(self)
            step = idx.step or 1
            idxs = range(start, stop, step)
            return [self[i] for i in idxs]

        if idx >= len(self):
            raise StopIteration

        # get position of record
        self.idx_file.seek(idx * self.idx_record_size + 4, 0)
        position = self.idx_file.read(self.idx_record_size)
        position = int.from_bytes(position, 'big', signed=False)

        # get record
        self.data_file.seek(position, 0)
        try:
            record = pickle.load(self.data_file)
        except Exception as e:
            print("Idx: {} - Position: {}".format(idx, position))
            raise

        # transform
        if self.transform:
            return self.transform(record)
        return record

    def __del__(self):
        self.idx_file.close()
        logger.info("Idx file closed ({})".format(self.data_path))
        self.data_file.close()
        logger.info("Data file closed ({})".format(self.data_path))

    def add_item(self, item: Any):
        self.idx_file.seek(0, 2)
        self.data_file.seek(0, 2)
        self.idx_file.write(self.data_file.tell().to_bytes(6, 'big', signed=False))
        pickle.dump(item, self.data_file)
        self.idx_file.seek(0, 0)
        num_records = int.from_bytes(self.idx_file.read(4), byteorder='big', signed=False)
        num_records += 1
        self.idx_file.seek(0, 0)
        self.idx_file.write(num_records.to_bytes(4, 'big', signed=False))
