# Visualizing the learned space-time attention 

This repository contains implementations of __Attention Rollout__ for __TimeSformer__ model. 

Attention Rollout was introduced in paper [Quantifying Attention Flow in Transformers](https://arxiv.org/abs/2005.00928). It is a method to use attention weights to understand how a self-attention network works, and provides valuable insights into which part of the input is the most important when generating the output. 


It assumes the attention weights determine the proportion of the incoming information that can propagate through the layers and we can use attention weights as an approximation of how information flow between layers. If `A` is a 2-D attention weight matrix at layer `l`, `A[i,j]` would represent the attention of token `i` at layer `l` to token `j` from layer `l-1`. And to compute the attention to the input tokens from the video, it recursively multiply the attention weights matrices, starting from the input layer up to layer `l`.

## Implementating Attention Rollout for TimeSformer

For divided space-time attention, each token has `2` dimensions,  let's denote the token as `z(p,t)`, where `p` is spatial dimension and  `t` is the time dimension; 

Each encoding block contains a <u>time attention layer</u> and a <u>space attention layer</u>. During __time attention__ block, each patch token only attends to patches at same spatial locations; During __space attention__, each patch only attends to the patches from same frame. If we use `T` and `S` to denote __time attention weights__ and __space attention weights__ respectively,`T[p,j,q]` would represent the attention of `z(p,j)` to `z(p,q)` from previous layer during time attention layer and `S[i,j,p]` would represent the space attention of `z(i,j)` to `z(p,j)` from time attention layer;

When we combined the space and time attention, each patch token will attends to all patches at every spatial locations from all frames (with the exception of the `cls_token`, we will discuss about it later) through an __unique path__. The attention path of `z(i,j)` to `z(p,q)` (where `p != 0`) is 
* space attention: `z(i,j)`-> `z(p,j)` 
* time attention: `z(p,j)`-> `z(p,q)`

we can calculate the combined space time attention for this layer as 
```python
W[i,j,p,q] = S[i,j,p]* T[p,j,q]
```

note that the classification token did not participate in the time attention layer - it was removed from the input before it enter the time attention layer and added back before passing to the space attention layer. This means it only attends to itself during time attention computation, we use an identity matrix to account for this. Since classification did not participate in time attention computation, all the tokens will only be able to attend to classification token from same frame, to address this limitation, in TimeSformer implementation, the `cls_token` output is averaged across all frames at end of each space-time attention block, so that it will be able to carry information from other frames, we also need to average its attention to all input tokens when we compute the combined space time attention

## Usage

Here is a notebook demostrate how to use attention rollout to visualize space time attention learnt from TimeSformer


[a colab notebook: Visualizing learned space time attention with attention rollout](https://colab.research.google.com/github/yiyixuxu/TimesFormer_rolled_attention/blob/main/visualizing_space_time_attention.ipynb)

## Visualizing the learned space time attention

This is the example used in the TimeSformer paper to demonstrate that the model can learn to attend to the
relevant regions in the video in order to perform complex spatiotemporal reasoning. we can see that
the model focuses on the configuration of the hand when visible and the object-only when not visible.
![alt text](https://github.com/yiyixuxu/TimesFormer_rolled_attention/blob/6f3bce9fdb35ab6178b15a27b1d7b493ae69d9aa/img.png?raw=true)
![alt text](https://github.com/yiyixuxu/TimesFormer_rolled_attention/blob/6f3bce9fdb35ab6178b15a27b1d7b493ae69d9aa/mask.png?raw=true)




## References
* papers:
  * [Quantifying Attention Flow inTransformers](https://arxiv.org/abs/2005.00928)
  * [Is Space-Time Attention All You Need for Video Understanding?](https://arxiv.org/pdf/2102.05095.pdf)
* code bases:
  * [TimeSformer](https://github.com/yiyixuxu/TimeSformer)
  * [Attention flow](https://github.com/samiraabnar/attention_flow#readme)
  * [vit-explain](https://github.com/jacobgil/vit-explain/blob/main/Readme.md)
