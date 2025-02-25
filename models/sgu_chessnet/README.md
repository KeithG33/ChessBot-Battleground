# Spatial Gating ChessNet
Implements spatial gating with attention (a-MLP) from the paper [Pay Attention to MLPs](https://arxiv.org/pdf/2105.08050v2)

## Architecture
The main block of the SGU ChessNet consists of two components:
1. A gMLP module: implementation is taken from the pseudocode of the paper, and uses the attention module suggested by the authors. However it is less tiny and uses 1D relative position bias for better performance.
2. Global MLP: a normal mlp with LayerNorm, GELU, linear layers, and residual connection between input and output. Acts on flattened features to attend globally


<div align="center"  id="image.png">
  <img src="image.png" style="width: 75%; height: auto;">
  <p><em>Pseudo code from paper. Original diagram modified to show tiny-attention module, used in the SGU ChessNet.</em></p>
</div>


```python
# Pseudo-code for SGU block as used in SGU ChessNet
def sgu_block(x):
  x = x + gmlp_block(x)
  x = x + mlp(x.flatten()).unflatten()
```

**Stats**:
- Layers: 24
- Hidden shape: (B, 64, 32)


### Training Plots and Scores
*Coming soon*
<!-- ![Training Plot](path_to_training_plot.png) -->

## Usage
Play against the model with the following command

```bash
chessbot play "sgu_chessnet" \
              --model-dir path_to_model \
              --model-weights path_to_weights

chessbot evaluate "sgu_chessnet" \
              --model-dir path_to_model \
              --model-weights path_to_weights
              --data-dir path_to_dataset
              --batch-sz 3072