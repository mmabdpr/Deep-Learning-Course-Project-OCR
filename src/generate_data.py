from pathlib import Path
import torch
from PIL import Image, ImageOps
import pandas as pd
import tqdm

from src.utils import label2char
from src.dataset import IdDataset


def generate_data():
    res_dir = Path('data/processed/dataset')

    ds = iter(IdDataset(128, 48))

    for i in range(1000):
        x, y = next(ds)
        torch.save(x, res_dir / f'{i}_x.torch')
        torch.save(x, res_dir / f'{i}_y.torch')


def generate_test_data():
    res_dir = Path('data/processed/evaluate')

    ds = iter(IdDataset(32, 48))

    labels = []
    images = []

    for i in tqdm.tqdm(range(1000)):
        x, y = next(ds)

        for img, label in zip(x, y):
            img = img.numpy()[0, :, :]
            img = (img + 0.5) * 255
            img = Image.fromarray(img).convert("L")
            label = label.numpy()
            label = ''.join([label2char[l] for l in label])
            images.append(img)
            labels.append(label)

    image_files = []
    for i, (img, label) in enumerate(zip(images, labels)):
        f = f'{i}_{label}.png'
        image_files.append((res_dir / f).resolve().as_posix())
        img.save(res_dir / f)

    df = pd.DataFrame({
        'label': labels,
        'file': image_files})

    df.to_json(res_dir / 'labels.json', indent=2, orient='records')


if __name__ == "__main__":
    # generate_data()
    generate_test_data()
