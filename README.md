# Visualizing the learned space-time attention 

This repository contain implementations of __Attention Rollout__ for __TimeSformer__ model. 

Attention Rollout is presented in paper [Quantifying Attention Flow inTransformers](https://arxiv.org/abs/2005.00928). It is a method to use attention weights to understand how a self-attention network works, and provides valuable insights into which part of the input is the most important when generating the output. 


It assumes the attention weights determine the proportion of the incoming information that can propagate through the layers and we can use attention weights as an approximation of how information flow between layers. If $\mathrm{A}$ is a 2D attention weight matrix at layer $i$, $\mathrm{A[i,j]}$ would represent the attention of $\mathrm{i_{th}}$  token to its  $\mathrm{j_{th}}$  input token. And to compute the attention to input tokens, it recursively multiply the attention weights matrices, starting from the input layer up to layer $i$.

## Implementating Attention Rollout for TimeSformer

For divided space-time attention, each token has `2` dimensions,  let's denote the output token of layer $l$ as  $\mathrm{z}_{(p,t)}^{l}$   where $\scriptsize p$ is spatial dimension and $\scriptsize t$ is the time dimension; 

Each encoding block contains a time attention layer and a space attention layer. During __time attention__ block, each patch only attends to patches at same spatial locations; During __space attention__, each patch only attends to the patches from same frame. If we use $\mathrm{T}$ and $\mathrm{S}$ to denote time attention weights and space attention weights respectively, $\mathrm{T_{i,j,q}}$ would represent the attention of $\mathrm{z}_{(i,j)}$ to $\mathrm{z}_{(i,q)}$ during time attention layer and $\mathrm{S_{i,j,k}}$ would represent the attention of $\mathrm{z}_{(i,j)}$ to $\mathrm{z}_{(k,j)}$ during space attention layer;

When we combined the space and time attention, each output token attends to each input token (except the `cls_token`) through an __unique path__. The attention path of $\mathrm{z}_{(i,j)}^{l}$ to $\mathrm{z}_{(p,q)}^{l-1}$ (where $\scriptsize k \neq 0$) is 
* space attention: $\mathrm{z}_{(i,j)} \rightarrow \mathrm{z}_{(k,j)}$ 
* time attention: $\mathrm{z}_{(k,j)}\rightarrow \mathrm{z}_{(k,q)}$

we can calculate the combined space time attention for this layer as $$\mathrm{W_{i,j,p,q}} = \mathrm{W\text{s}_{i,j,k}} \cdot \mathrm{W\text{t}_{k,j,q}}$$

note that the classification token did not participate in the time attention layer - it was removed from the input to time attention and added back to its output before passing to the space attention layer. This means it only attends to itself during time attention computation, we use an identity matrix to account for this. Since classification did not participate in time attention computation, all the tokens will only be able to attend to classification token from same time dimension, to address this, in TimeSformer implementation, the `cls_token` output is averaged across all frames at end of each time space attention block, so that the `cls_token` at each frame would carry our understanding from other frames, we also need to average its attention to all input tokens when we compute the combined space time attention

## Usage
colab notebook

## visualizing the learned space time attention

this is the example used in the TimeSformer paper to demonstrate that it learns to attend to the
relevant regions in the video in order to perform complex
spatiotemporal reasoning. we can observe that
the model focuses on the configuration of the hand when
visible and the object-only when not visible.




## references
* papers:
  * [Quantifying Attention Flow inTransformers](https://arxiv.org/abs/2005.00928)
  * [Is Space-Time Attention All You Need for Video Understanding?](https://arxiv.org/pdf/2102.05095.pdf)
* code bases:
  * [Attention flow](https://github.com/samiraabnar/attention_flow#readme)
  * [vit-explain](https://github.com/jacobgil/vit-explain/blob/main/Readme.md)
