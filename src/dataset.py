import random
import torch
from src.credit_id_dataset import CreditIdDataset
from src.national_id_dataset import NationalIdDataset


class IdDataset(torch.utils.data.IterableDataset):
    def __init__(self, batch_size, output_height):
        super(IdDataset).__init__()
        self.batch_size = batch_size
        self.output_height = int(output_height)
        self.credit_dataset = iter(CreditIdDataset(batch_size, output_height))
        self.national_dataset = iter(NationalIdDataset(batch_size, output_height))

    def __iter__(self):
        while True:
            if random.random() > 0.5:
                yield next(self.credit_dataset)
            else:
                yield next(self.national_dataset)
                
                
if __name__ == "__main__":
    ds = IdDataset(32, 48)
    it = iter(ds)
    X, Y = next(it)
    X2, Y2 = next(it)
    pass
