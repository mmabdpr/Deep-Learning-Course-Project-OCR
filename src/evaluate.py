from pathlib import Path
import pandas as pd
import numpy as np
import torch
from PIL import Image, ImageOps
from time import perf_counter_ns
import json
import tqdm

from src.crnn import CRNN48
from src.decoder import ctc_decode
from src.metrics import calculate_accuracy_str

from src.utils import id_label, label2char


def load_model_for_evaluation(checkpoint_path: Path) -> torch.nn.Module:
    model = CRNN48()
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model


def load_image(img_path: Path) -> torch.Tensor:
    img = Image.open(img_path)
    img = ImageOps.grayscale(img)
    img_height = 48.
    img_width = img_height / img.height * img.width
    img = img.resize((int(img_width), int(img_height)))
    img = np.array(img) / 255. - 0.5
    img = np.expand_dims(img, 0)
    img = torch.tensor(img, dtype=torch.float32)
    return img


def text_to_label(txt) -> torch.Tensor:
    label = id_label(txt)
    label = torch.tensor(label, dtype=torch.int32)
    return label


def evaluate_model(model: torch.nn.Module, data_x: list[torch.Tensor], data_y: list[str]):
    results = []
    timings = []
    for x in tqdm.tqdm(data_x):
        x = torch.unsqueeze(x, 0)

        with torch.no_grad():
            t1 = perf_counter_ns()
            pred = model(x)
            t2 = perf_counter_ns()
            res = ctc_decode(pred, label2char=label2char, beam_size=10)
            t3 = perf_counter_ns()
            results.append(''.join(res[0]))
            timings.append((t2 - t1, t3 - t2))

    accuracy = calculate_accuracy_str(data_y, results)

    return {
        'accuracy': accuracy,
        'avg_model_runtime_ns': np.average([t[0] for t in timings]),
        'avg_decode_runtime_ns': np.average([t[1] for t in timings]),
    }


def save_evaluation_results(res, path):
    with open(path / 'eval_results.json', 'w') as f:
        json.dump(res, f)


def run_evaluation(checkpoint: Path, labels: Path):
    df = pd.read_json(labels, orient='records')
    data_x = [load_image(img) for img in df['file'].tolist()]
    data_y = [str(t) for t in df['label'].tolist()]
    model = load_model_for_evaluation(checkpoint)
    results = evaluate_model(model, data_x, data_y)
    save_evaluation_results(results, labels.resolve().parent)


def main():
    eval_labels = Path('data/raw/evaluate/labels.json')
    # eval_labels = Path('data/processed/evaluate/labels.json')
    checkpoint = Path('data/checkpoints/40001.torch')
    run_evaluation(checkpoint, eval_labels)


if __name__ == "__main__":
    main()
