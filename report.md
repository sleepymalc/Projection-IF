# Projection-Regularization for Data Attribution at Scale

Main theorem: the sketched scores $\widetilde{\phi}_{\lambda}(g_i, g_{\text{test}})$ approximate the exact scores $\phi_{\lambda}(g_i, g_{\text{test}})$ within $(1±\epsilon)$-bounds, where
$$
\phi_{\lambda}(g_i, g_{\text{test}})
= g_i^{\top} (F+\lambda I)^{-1} g_{\text{test}},\quad
\widetilde{\phi}_{\lambda}(g_i, g_{\text{test}})
= (Pg_i)^{\top} (PFP^{\top} +\lambda I)^{-1} (Pg_{\text{test}}),
$$
given that $P\in \mathbb{R}^{m \times k}$ is a standard oblivious sketch with $m = \widetilde{\Omega}(d_{\lambda}(F) / \epsilon^2)$.

>[!Note]
>
>**Theorem 2** only guarantees $\widetilde{\phi}_{\lambda}(g_i, g_{\text{test}}) \in (1\pm \epsilon) \phi_{\lambda}(g_i, g_{\text{test}})$ for $g_{\text{test}} \in \operatorname{range}(F)$. In reality, this is rarely the case.

## Experiments

Due to the scale, we utilize SJLT (sparse Johnson-Lindenstrauss transform).

| Model   | Dataset | Parameters | Train Samples |
| ------- | ------- | ---------- | ------------- |
| lr      | mnist   | 7,850      | 5,000         |
| mlp     | mnist   | 109,386    | 5,000         |
| resnet9 | cifar2  | 4,825,154  | 5,000         |

### Spectrum Bounds Validation

We first investigate the spectrum sandwich bound. The experimental procedure is:

1. Compute the empirical Fisher matrix $F = \frac{1}{n}\sum_{i=1}^{n} g_i g_i^{\top}$ and its eigenspectrum
2. For each $\lambda \in \{10^{-3}, 10^{-2}, 10^{-1}, 1, 10, 100, 1000\}$, compute $d_{\lambda}(F) = \mathrm{tr}(F(F+\lambda I)^{-1})$
3. For each $(m, \lambda)$ pair, compute both exact scores $\phi_{\lambda}$ and sketched scores $\widetilde{\phi}_{\lambda}$
4. Measure the empirical approximation error $\epsilon = |\widetilde{\phi}_{\lambda} / \phi_{\lambda} - 1|$

We plot the following:

1. **$m / d_{\lambda}$ vs. $\epsilon$**: We expect this to follow a straight line with slope $-1/2$ in log-log plot, validating $\epsilon \propto 1/\sqrt{m/d_{\lambda}}$.
2. **$m/d_{\lambda}$ vs. $\widetilde{\phi}_{\lambda} / \phi_{\lambda}$**: As $m/d_{\lambda}$ grows, the ratio should converge to $1$.
3. **Spectrum of $F$**: Shows eigenvalue decay, which determines $d_{\lambda}$ for different $\lambda$.
4. **$\lambda$ vs. $d_{\lambda} / \operatorname{rank}(F)$**: Measures how fast $d_{\lambda}$ shrinks as $\lambda$ grows (compression ratio).

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
