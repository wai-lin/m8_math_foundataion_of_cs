# 📝 Calculus 101 Cheat Sheet

### 1. Limits: "Approaching"

Determines the value a function approaches as input gets closer to a specific point (or infinity).

**Notation:** $\lim_{x \to c} f(x) = L$

| Type | Intuition | Example |
|---|---|---|
| Plug-in | Usually, just plug the number in. | $\lim_{x \to 2} (3x+1) = 7$ |
| Infinity | As $x$ grows forever, does $f(x)$ settle? | $\lim_{x \to \infty} \frac{1}{x} = 0$ (1 pizza, $\infty$ people) |
| 0/0 Case | "Hole" in graph. Simplify algebra first. | $\frac{x^2-1}{x-1} \to (x+1) \to 2$ |

---

### 2. Derivatives: "Rate of Change"

Finds the **Slope** of the curve at a specific instant.

**Notation:** $f'(x)$ or $\frac{dy}{dx}$

**The Golden Rule (Power Rule):**

$$f(x) = x^n \quad \Rightarrow \quad f'(x) = n \cdot x^{n-1}$$

| Function f(x) | Derivative f′(x) | Note |
|---|---|---|
| Constant ($5$) | $0$ | Flat line = 0 slope |
| Linear ($x$) | $1$ | Slope is constant |
| Quadratic ($x^2$) | $2x$ | Slope changes |
| Cubic ($x^3$) | $3x^2$ |  |
| $\sin(x)$ | $\cos(x)$ |  |
| $e^x$ | $e^x$ | The "indestructible" function |

---

### 3. Integrals: "Accumulation"

Finds the **Area** under the curve. The reverse of a derivative.

**Notation:** $\int f(x) dx$

**The Golden Rule (Reverse Power Rule):**

$$\int x^n dx = \frac{x^{n+1}}{n+1} + C$$

**Types:**
- Indefinite Integral (General): $\int 2x \, dx = x^2 + C$
  - Result is a formula. Don't forget $+C$!
- Definite Integral (Exact Area): $\int_{0}^{3} 2x \, dx = 9$
  - Result is a number. Subtract bottom from top: $[F(b) - F(a)]$.

---

### ⚡ The "Big 3" Relationship

1. **Position** ($s(t)$) $\xrightarrow{\text{Derivative}}$ **Velocity** ($v(t)$)
2. **Velocity** ($v(t)$) $\xrightarrow{\text{Derivative}}$ **Acceleration** ($a(t)$)
3. **Acceleration** ($a(t)$) $\xrightarrow{\text{Integral}}$ **Velocity** ($v(t)$)

**Fundamental Theorem of Calculus:**

Differentiation and Integration are opposites.

$\int_{a}^{b} f'(x) \, dx = f(b) - f(a)$

---

### 📌 Quick Convergence Reference

- **Linear Convergence ($r=1$):** Error is cut by a constant ratio (e.g., $e_{n+1} = 0.5 \cdot e_n$). Steady.
- **Quadratic Convergence ($r=2$):** Error is squared ($e_{n+1} \approx e_n^2$). Explodes in accuracy ($0.01 \to 0.0001$). Fast.
