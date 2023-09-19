# Linear classification

In classification task we have categorical targets $y \in \mathcal Y = \{1 ,\ldots, K\}$.

## Why not linear regression?

However, the class labels in this settings are numbers, and we could fit a linear regression. Why is this not a good idea?

```{admonition} Answer
:class: dropdown
The possible problems are:
* Inappropriate predictions: $\widehat y$ could easily be outside $\mathcal Y$
* No inherent ordering between class labels
* Loss function mismatch: MSE could be quite poor metric of quality
```

## Binary case

Suppose that $\mathcal Y = \{-1, 1\}$, then we could predict labels of $\boldsymbol x \in \mathbb R^d$ by the formula

$$
    \widehat y = \mathrm{sgn}\Big(\sum\limits_{j=1}^d x_j w_j\Big) = \mathrm{sgn}(\boldsymbol x^\top \boldsymbol w).
$$

What about loss function? Rewrite the misclassification rate {eq}`mis-rate` as

```{math}
:label: mis-rate-class
\mathcal L(\boldsymbol w) = \sum\limits_{i=1}^n [\boldsymbol x_i^\top \boldsymbol w y_i < 0].
```

### Margins

Define the **margin** of the training sample $(\boldsymbol x_i, y_i)$ as

$$
    M_i = \boldsymbol x_i^\top \boldsymbol w y_i.
$$

If the margin is positive, the prediction is correct, and vise versa. Now we can express the loss function {eq}`mis-rate-class` in terms of margins:

```{math}
:label: mis-rate-margin
\mathcal L(\boldsymbol w) = \frac 1n\sum\limits_{i=1}^n \ell(M_i), \quad \ell(M) = [M < 0].
```

The function $\ell$ is discontinuous, which impedes the direct optimization of this loss functions. That's why $\ell$ is often substituted by some smooth loss function $\overline{\ell}(M)$, which estimates $\ell(M)$ from above: $\ell(M) \leqslant \overline{\ell}(M)$.

Popular choices of $\overline{\ell}(M)$:

- $V(M) = (1 - M)_+$ (SVM)
- $H(M) = (-M)_+$ (Hebb's rule)
- $L(M) = \log_2(1 + e^{-M})$ (LR)
- $Q(M) = (1 - M)^2$ (quadratic)
- $S(M) = \frac 2{1 + e^{-M}}$ (sigmoid)
- $E(M) = e^{-M}$ (exponential)
