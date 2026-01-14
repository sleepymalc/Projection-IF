# Projection-Regularization for Data Attribution at Scale

**Main Theorem.** Let $P \in \mathbb{R}^{m \times d}$ be a sketching matrix whose rows are i.i.d. sub-Gaussian random vectors. For any $\epsilon, \delta \in (0,1)$, if the sketch size satisfies
$$
m = \Omega\left(\epsilon^{-2}(d_\lambda(F) + \log(1/\delta))\right),
$$
then with probability at least $1-\delta$, the following bounds hold for all $g, g' \in \operatorname{range}(F)$:

1. **Self-influence bound (Eq. 1)**:
$$
|\widetilde{\phi}_\lambda(g) - \phi_\lambda(g)| \leq \epsilon \cdot \phi(g)
$$

2. **Bilinear form bound (Eq. 2)**:
$$
|\widetilde{B}_\lambda(g, g') - B_\lambda(g, g')| \leq \epsilon \sqrt{\phi_\lambda(g)} \sqrt{\phi_\lambda(g')}
$$

Here, $\phi_\lambda(g) := g^\top(F+\lambda I)^{-1}g$ and $\widetilde{\phi}_\lambda(g) := (Pg)^\top(PFP^\top + \lambda I)^{-1}(Pg)$, with $B_\lambda$ and $\widetilde{B}_\lambda$ denoting the corresponding bilinear forms. The effective dimension is $d_\lambda(F) = \mathrm{tr}(F(F+\lambda I)^{-1})$.

>[!Note]
>
>The theorem only guarantees bounds for $g, g' \in \operatorname{range}(F)$. For test gradients $g' \notin \operatorname{range}(F)$, Section 3 of the theoretical document provides leakage analysis showing that the approximation error for the kernel component decays at rate $O(m^{-1/2})$.

## Experiments

Due to the scale, we utilize SJLT (sparse Johnson-Lindenstrauss transform).

| Model   | Dataset | Parameters | Train Samples |
| ------- | ------- | ---------- | ------------- |
| lr      | mnist   | 7,850      | 5,000         |
| mlp     | mnist   | 109,386    | 5,000         |
| resnet9 | cifar2  | 4,825,154  | 5,000         |

### Spectrum Bounds Validation

We validate the theoretical bounds from the Main Theorem. The experimental procedure is:

1. Compute the empirical Fisher matrix $F = \sum_{i=1}^{n} g_i g_i^{\top}$ and its eigenspectrum
2. For each $\lambda$, compute $d_{\lambda}(F) = \mathrm{tr}(F(F+\lambda I)^{-1})$
3. For each $(m, \lambda)$ pair, compute both exact and sketched scores
4. Measure the empirical approximation error (metric depends on test mode)

#### Test Gradient Selection

The experiment supports two modes:

- **Self-influence mode** (`--test_mode self`): Validates Eq. 1 using training gradients as test vectors, computing diagonal self-scores $\phi_\lambda(g_i, g_i)$. Error metric: $|\widetilde{\phi}_\lambda - \phi_\lambda| / \phi_\lambda$ (ratio deviation).

- **Test mode** (`--test_mode test`): Validates Eq. 2 using held-out test set gradients, computing the full cross-score matrix $B_\lambda(g_{\text{train}}, g_{\text{test}})$ of shape $(n_{\text{train}}, k_{\text{test}})$. Error metric: $|\widetilde{B}_\lambda - B_\lambda| / (\sqrt{\phi_\lambda(g)} \cdot \sqrt{\phi_\lambda(v)})$ (normalized bilinear form error).

We plot the following (2x2 layout):

1. **Self-Influence Error (Top-Left)**: $m/d_\lambda$ vs. $|\widetilde{\phi}_\lambda/\phi_\lambda - 1|$ (95th percentile). Validates Eq. 1.
2. **Bilinear Form Error (Top-Right)**: $m/d_\lambda$ vs. normalized error $|\widetilde{B}_\lambda (g, g')-B_\lambda (g, g')|/(\sqrt{\phi_\lambda(g)}\sqrt{\phi_\lambda(g')})$ (95th percentile). Validates Eq. 2.
3. **Eigenvalue Spectrum (Bottom-Left)**: Shows eigenvalue decay of $F$, which determines $d_\lambda$ for different $\lambda$.
4. **Compression Ratio (Bottom-Right)**: $\lambda$ vs. $d_\lambda / \operatorname{rank}(F)$, showing how regularization compresses the effective dimension.

Both error plots should follow $\epsilon \propto 1/\sqrt{m/d_\lambda}$ (slope $-1/2$ in log-log scale).

#### MNIST+Logistic Regression

- Parameters: 7,850, Rank: 4,525
- Effective dimensions: $d_{0.1} = 704$, $d_{1} = 199$, $d_{10} = 40$

![Spectrum Bounds - MNIST LR](experiments/results/figures/spectrum_mnist_lr.png)

#### MNIST+MLP

- Parameters: 109,386, Rank: 5,000
- Effective dimensions: $d_{0.01} = 815$, $d_{0.1} = 160$, $d_{1} = 24$

![Spectrum Bounds - MNIST MLP](experiments/results/figures/spectrum_mnist_mlp.png)

#### CIFAR2+ResNet9

- Parameters: 4,825,154, Rank: 4,560
- Effective dimensions: $d_{0.001} = 34$, $d_{0.01} = 16$, $d_{0.1} = 5$

![Spectrum Bounds - CIFAR2 ResNet9](experiments/results/figures/spectrum_cifar2_resnet9.png)

### Hyperparameter Selection

Next, we investigate how to choose $m$ in practice. The experimental procedure is:

1. **Step 1 (λ selection)**: Fix $m$ large, sweep $\lambda$ to maximize LDS on a validation set, obtaining $\lambda^{\ast}$
2. **Step 2 (m sweep)**: Fix $\lambda = \lambda^{\ast}$, sweep $m$ and measure LDS to find the minimum $m$ achieving near-optimal performance

We use a 90/10 split of the test set into validation and held-out test sets. The validation set is used for hyperparameter selection, and the test set provides final evaluation.

We plot the following:

1. **$\lambda$ vs. LDS (Val Set)**: Shows the λ sweep with the best λ* marked, along with the held-out test LDS at λ*
2. **$m$ vs. LDS (Val Set)**: Shows how LDS varies with projection dimension, with the empirical constant $c_{95\%} = m_{95\%} / d_{\lambda^{\ast}}$
3. **$\lambda$ vs. $d_{\lambda}$**: Shows effective dimension as a function of regularization
4. **$c = m/d_{\lambda^{\ast}}$ vs. Normalized LDS**: Directly shows the scaling factor $c$ needed to achieve a given fraction of optimal LDS

#### MNIST+Logistic Regression

- Best $\lambda^{\ast} = 1.0$, corresponding $d_{\lambda^{\ast}} = 200$
- Val LDS = 0.360, Test LDS = 0.380
- **Empirical constant**: $c_{95\%} = 4096 / 200 = 20.5$

![Hyperparam Selection - MNIST LR](experiments/results/figures/hyperparam_mnist_lr.png)

#### MNIST+MLP

- Best $\lambda^{\ast} = 0.01$, corresponding $d_{\lambda^{\ast}} = 742$
- Val LDS = 0.187, Test LDS = 0.181
- **Empirical constant**: $c_{95\%} = 8192 / 742 = 11.0$

![Hyperparam Selection - MNIST MLP](experiments/results/figures/hyperparam_mnist_mlp.png)

#### CIFAR2+ResNet9

- Best $\lambda^{\ast} = 10^{-5}$, corresponding $d_{\lambda^{\ast}} = 277$
- Val LDS = 0.185, Test LDS = 0.187
- **Empirical constant**: $c_{95\%} = 2048 / 277 = 7.4$

![Hyperparam Selection - CIFAR2 ResNet9](experiments/results/figures/hyperparam_cifar2_resnet9.png)

#### Summary: Practical Rule for Choosing $m$

| Model   | Dataset | $d_{\lambda^{\ast}}$ | $m_{95\%}$ | $c_{95\%}$ |
| ------- | ------- | -------------------- | ---------- | ---------- |
| lr      | mnist   | 200                  | 4,096      | 20.5       |
| mlp     | mnist   | 742                  | 8,192      | 11.0       |
| resnet9 | cifar2  | 277                  | 2,048      | 7.4        |

**Average $c_{95\%} \approx 13$**

Hence, To achieve 95% of optimal LDS, set $m \approx 10$-$20 \times d_{\lambda}$. This provides a simple guideline: once $\lambda^{\ast}$ is selected, compute $d_{\lambda^{\ast}}$ and set $m = c \cdot d_{\lambda^{\ast}}$ where $c \in [10, 20]$ depending on desired accuracy.
