
# Use of TransMLA for Qwen3 Models

The [TransMLA](https://arxiv.org/pdf/2502.07864) framework was proposed to enable conversion of language models trained using Grouped Query Attention (GQA) to the DeepSeek Multi-head Latent Attention (MLA) architecture.

As explained in this [Github issue](https://github.com/MuLabPKU/TransMLA/issues/38), the current version of the TransMLA framework does not support Qwen3 models. The purpose of this document is to explain the reason for this from a mathematical perspective and expore some potential solutions.

## Explanation of problem

A simple example with the following parameter settings will be used to explain the problem:

- `n_head = 16`
