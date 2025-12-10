
# Use of TransMLA for Qwen3 Models

The [TransMLA](https://arxiv.org/pdf/2502.07864) framework was proposed to enable conversion of language models trained using Grouped Query Attention (GQA) to the DeepSeek Multi-head Latent Attention (MLA) architecture.

As explained in this [Github issue](https://github.com/MuLabPKU/TransMLA/issues/38), the current version of the TransMLA framework does not support Qwen3 models. The purpose of this document is to explain the reason for this from a mathematical perspective and expore some potential solutions.

## Explanation of problem

A simple example with the following parameter settings will be used to explain the problem:

- Size of hidden dimension per key-value head = $d$ = 4
- Number of key and value heads = $g$ = 2. Note that all key and value heads are merged into a single head for the Multi-Query Attention (MQA) mode of MLA. 
- Number of query heads = $h$ = 8. This number is actually irrelevant for the subsequent discussion.
- `freqfold` parameter = $\phi$ = 2

Consider the dot product of query and key in 3 different scenarios:

### No RMSNorm

The hidden dimension of each query vector is up-projected from $d = 4$ to $gd = 8$ before performing the dot product with the key vector. In the MQA mode, there will be $h$ query vectors and a single key vector. All these vectors will have dimension $gd$. The dot product is given by:


$$
[q_1, q_2, q_3, q_4, q_5, q_6, q_7, q_8]^R \cdot [k_1, k_2, k_3, k_4, k_5, k_6, k_7, k_8]^R 
$$

, where the superscript 'R' denotes [Rotary Position Embeddings](https://arxiv.org/pdf/2104.09864). Let $q_{2l - 1}$ be the real components of the query vectors and $q_{2l}$ be the imaginary components, where $1 \le l \le 4$. In the subsequent discussion, the real and imaginary components will be grouped together. Hence, the dot product can be written as:

$$
([q_1, q_3, q_5, q_7]; [q_2, q_4, q_6, q_8])^R \cdot ([k_1, k_3, k_5, k_7]; [k_2, k_4, k_6, k_8])^R
$$

The proof in Appendix B of the [TransMLA paper](https://arxiv.org/pdf/2502.07864) shows that one can apply a rotation matrix $U$ to the real and imaginary components of both the query and key vectors without changing the magnitude of the dot product. For the case $\phi = 2$, the dot product becomes 

$$
(U[q_1, q_3, q_5, q_7]; U[q_2, q_4, q_6, q_8])^R \cdot (U[k_1, k_3, k_5, k_7]; U[k_2, k_4, k_6, k_8])^R
$$

Apparently, the same rotation matrix must be applied to both the odd and even components although it is not clear why.


### With single RMSNorm operation applied across all key-value heads

The [RMSNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html) operation for queries and keys is given by 

$$
\tilde{q}_i = \alpha_j \frac{q_i}{ \sqrt{\sum_{i = 1}^{gd} q_i^2} } \quad \text{and} \quad \tilde{k}_i = \beta \frac{k_i}{ \sqrt{\sum_{i = 1}^{gd} k_i^2}}
$$

, where

$$
j=
\begin{cases}
i \text{mod } d, & \text{if } i \text{mod } d > 0,\\
d, & \text{otherwise } .
\end{cases}
$$

As demonstrated [here](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/11_qwen3/standalone-qwen3.ipynb), Qwen3 applies the RMSNorm operation before performing RoPE.

Let $q_n = \sum_{i = 1}^{hd} q_i^2$ and $k_n = \sum_{i = 1}^{gd} k_i^2$. The dot product becomes


$$
(\frac{1}{q_n}[\alpha_1 q_1, \alpha_3 q_3, \alpha_1 q_5, \alpha_3 q_7]; \frac{1}{q_n} [ \alpha_2 q_2, \alpha_4 q_4, \alpha_2 q_6, \alpha_4 q_8])^R \cdot ( \frac{1}{k_n} [ \beta_1 k_1, \beta_3 k_3, \beta_1 k_5, \beta_3 k_7]; \frac{1}{k_n} [\beta_2 k_2, \beta_4 k_4, \beta_2 k_6, \beta_4 k_8])^R
$$

Upon performing rotation of queries and keys, one obtains


$$
(\frac{1}{q_n}U[\alpha_1 q_1, \alpha_3 q_3, \alpha_1 q_5, \alpha_3 q_7]; \frac{1}{q_n} U[ \alpha_2 q_2, \alpha_4 q_4, \alpha_2 q_6, \alpha_4 q_8])^R \cdot ( \frac{1}{k_n} U[ \beta_1 k_1, \beta_3 k_3, \beta_1 k_5, \beta_3 k_7]; \frac{1}{k_n} U[\beta_2 k_2, \beta_4 k_4, \beta_2 k_6, \beta_4 k_8])^R
$$

The different weights used for the various channels means that the RMSNorm operation must be applied after the up-projection query vectors from dimension $d$ to $gd$ and before the rotation is applied. This is required to ensure that the magnitude of the dot product is preserved after the inclusion of the rotation operation. This means that the rotation operation cannot be fused with up-projection as a single matrix multiplication, implying that the weights for $U$ must be updated separately from those used for the up-projection during training. This causes a problem because it may not be easy to update the weights for $U$ while imposing the constraint that it remains an orthogonal matrix.

If it can be assumed that the values of RMSNorm scaling parameters are similar for adjacent channels (i.e. $\alpha = \alpha_1 \approx \alpha_2 \approx \alpha_3 \approx \alpha_4$, $\beta = \beta_1 \approx \beta_2 \approx \beta_3 \approx \beta_4$), this problem can be avoided. In this case, one can write

$$
\frac{\alpha}{q_n}(U[ q_1, q_3, q_5, q_7]; U[ q_2, q_4, q_6, q_8])^R \cdot \frac{\beta}{k_n} ( U[ k_1, k_3, k_5, k_7]; U[ k_2, k_4, k_6, k_8])^R
$$


If a single weight is used for all channels, this problem can be avoided. 
