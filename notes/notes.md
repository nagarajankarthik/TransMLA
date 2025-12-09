
# Use of TransMLA for Qwen3 Models

The [TransMLA](https://arxiv.org/pdf/2502.07864) framework was proposed to enable conversion of language models trained using Grouped Query Attention (GQA) to the DeepSeek Multi-head Latent Attention (MLA) architecture.

As explained in this [Github issue](https://github.com/MuLabPKU/TransMLA/issues/38), the current version of the TransMLA framework does not support Qwen3 models. The purpose of this document is to explain the reason for this from a mathematical perspective and expore some potential solutions.

## Explanation of problem

A simple example with the following parameter settings will be used to explain the problem:

- Size of hidden dimension per head = $d$ = 4
- Number of key and value heads = $g$ = 2. Note that all key and value heads are merged into a single head for the Multi-Query Attention (MQA) mode of MLA. 
- Number of query heads = $h$ = 8. This number is actually irrelevant for the subsequent discussion.
- `freqfold` parameter = $\phi$ = 2

Consider the dot product of query and key in 3 different scenarios:

### No RMSNorm

The hidden dimension of each query head is up-projected from $d = 4$ to $gd = 8$ before performing the dot product with the key vector. In the MQA mode, there will be $h$ query vectors and a single key vector. All these vectors will have dimension $gd$. The dot product is given by:


$$
[q_1, q_2, q_3, q_4, q_5, q_6, q_7, q_8]^R \cdot [k_1, k_2, k_3, k_4, k_5, k_6, k_7, k_8]^R 
$$

, where the superscript 'R' denotes [Rotary Position Embeddings](https://arxiv.org/pdf/2104.09864). Let $q_{2l - 1}$ be the real components of the query vectors and $q_{2l}$ be the imaginary components, where $1 \le l \le 4$. In the subsequent discussion, the real and imaginary components will be grouped together. Hence, the dot product can be written as:

$$
[q_1, q_3, q_5, q_7, q_2, q_4, q_6, q_8]^R \cdot [k_1, k_3, k_5, k_7, k_2, k_4, k_6, k_8]^R
$$

The proof in Appendix B of the [TransMLA paper](https://arxiv.org/pdf/2502.07864) shows that one can apply a rotation matrix $U$ to the real and imaginary components of both the query and key vectors without changing the magnitude of the dot product. For the case $\phi = 2$, the dot product becomes 

$$
(U[q_1, q_3, q_5, q_7]; U[q_2, q_4, q_6, q_8])^R \cdot (U[k_1, k_3, k_5, k_7]; U[k_2, k_4, k_6, k_8])^R
$$

Apparently, the same rotation matrix must be applied to both the odd and even components although it is not clear why.


### With single RMSNorm operation applied across all heads

The [RMSNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html) operation for queries and keys is given by 

$$
\tilde{q}_i = \gamma_j q_i / \sqrt{\sum_{i = 1}^{gd} q_i^2} \quad \text{and} \quad \tilde{k}_i = \gamma_j k_i / \sqrt{\sum_{i = 1}^{gd} k_i^2}
$$

, where

$$
j=
\begin{cases}
j mod d, & \text{if } j mod d > 0,\\
d, & \text{if } otherwise.
\end{cases}
$$

