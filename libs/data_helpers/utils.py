import os
import pickle

from tqdm import tqdm
from typing import List, Any


def create_bytedataset(output_dir, data: List[Any]):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    idx_writer = open(os.path.join(output_dir, "idxs.pkl"), "wb")
    dataset_size_place_holder = (0).to_bytes(4, 'big', signed=False)
    idx_writer.write(dataset_size_place_holder)

    data_writer = open(os.path.join(output_dir, "data.pkl"), "wb")
    num_records = 0
    for item in tqdm(data):
        idx_writer.write(data_writer.tell().to_bytes(6, 'big', signed=False))
        pickle.dump(item, data_writer)
        num_records += 1

    data_writer.close()
    idx_writer.seek(0, 0)
    idx_writer.write(num_records.to_bytes(4, 'big', signed=False))


class ByteDatasetWriter:
    def __init__(self, data_path: str, idx_record_size: int = 6):
        self.data_path = data_path
        self.idx_record_size = idx_record_size
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        idx_file_path = os.path.join(data_path, "idxs.pkl")
        data_file_path = os.path.join(data_path, "data.pkl")
        if not os.path.exists(idx_file_path) or not os.path.exists(data_file_path):
            with open(idx_file_path, 'w'), open(data_file_path, 'w'):
                pass
        with open(idx_file_path, 'ab') as f:
            if f.tell() == 0:
                f.write((0).to_bytes(4, 'big', signed=False))

    def add_item(self, item: Any):
        with open(os.path.join(self.data_path, "idxs.pkl"), "rb+") as idx_file, \
                open(os.path.join(self.data_path, "data.pkl"), "rb+") as data_file:
            idx_file.seek(0, 2)
            data_file.seek(0, 2)
            idx_file.write(data_file.tell().to_bytes(self.idx_record_size, 'big', signed=False))
            pickle.dump(item, data_file)
            idx_file.seek(0, 0)
            num_records = int.from_bytes(idx_file.read(4), byteorder='big', signed=False)
            num_records += 1
            idx_file.seek(0, 0)
            idx_file.write(num_records.to_bytes(4, 'big', signed=False))
