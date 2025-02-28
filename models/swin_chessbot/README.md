# Swin Transformer Chess Bot
Swin Transformer and some prediction heads for chess
## Architecture
The feature extractor is a Swin Transformer from the `timm` library, so check out the [`timm` repo](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/swin_transformer.py) or check out [the paper](https://arxiv.org/abs/2103.14030). 

The two prediction networks contain 4 linear layers, LayerNorm+GELU activation, and a skip connection between the first two layers.

### Training Plots and Scores
*Coming soon*
<!-- ![Training Plot](path_to_training_plot.png) -->

<div align="center">

| Model Name   | Layers | Model Shape  | Params      | Weights       |
|--------------|--------|--------------|-------------|---------------|
| Swin-Transformer ChessBot | 20     | (B, 64, 24)  | 190M        | [Download Coming](path_to_model) |

</div>

## Usage
*Coming Soon* - 
Play against the model with the following command

```bash
chessbot play "swin_chessbot" \
              --model-dir path_to_model \
              --model-weights path_to_weights

chessbot evaluate "swin_chessbot" \
              --model-dir path_to_model \
              --model-weights path_to_weights
              --data-dir path_to_dataset
              --batch-sz 3072