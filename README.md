## Usage

### Evaluation

To run the saved model on custom image data you have to follow these steps:
- Train the model and save a checkpoint with the following format:
  - A dictionary that has at least one key named `model` which points to the state_dict of the pytorch model (`CRNN48`)
- Alternatively, you can download a trained model from [this link](https://drive.google.com/file/d/1Py8VrMkoX2qPtVRAM3MglZLmwarQUY-u/view?usp=sharing).
- Place your images in a directory.
- In the same directory create a json file named `labels.json` with the following format:
   ```json
    [
        {
            "file": "path/to/an/image.png",
            "label": "12359129"
        },
        {
            "file": "path/to/another/image.png",
            "label": "1235123159"
        }
    ]
    ```
- Edit the path variables in `evaluate.py` accordingly.
    - `eval_labels` should point to the `labels.json`.
    - `checkpoint` should point to the checkpoint file.
- Run `src.evaluate` module.
- Results are saved in a file named `eval_results.json` besides the `labels.json`.

### Notes

- If any of the images are too narrow in width, an exception will occur.