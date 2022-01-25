from src.dataset import IdDataset
from pathlib import Path
import torch


def generate_data():
    res_dir = Path('data/processed/dataset')
    
    ds = iter(IdDataset(128, 48))
    
    for i in range(1000):
        x, y = next(ds)
        torch.save(x, res_dir / f'{i}_x.torch')
        torch.save(x, res_dir / f'{i}_y.torch')


if __name__ == "__main__":
    generate_data()