# Model based on T3Set
We build a simple model based on T3Set to validate the usability of T3Set.

- CUDA Version: `12.2`
- Python Version: `3.11`

Other packages please refer to `requirements.txt`
## Example Usage
```bash
python train.py --use_lstm_layer False --use_learnable_mapping True --top_k 9
```
- use `use_lstm_layer` to control the usage of LSTM layer
- use `use_learnable_mapping` to control the usage of learnable mapping
- use `top_k` to control the number of top k suggestions

We provide a weight file for the model in `./weights`. Please specify the params in `settings.py`.
More detail usage (e.g. loading model) could be found in `scripts/eval_scrpits`.
## License
CC BY-NC-SA 4.0