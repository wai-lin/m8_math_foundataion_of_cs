# Gradient Descent Visualizer 🎯

An interactive web application that visualizes and compares different gradient descent optimization algorithms for linear regression.

## Features

### 📊 Three Optimization Algorithms
1. **Vanilla (Batch) Gradient Descent** - Uses entire dataset for each update
2. **Stochastic Gradient Descent (SGD)** - Updates using individual samples
3. **Mini-batch Gradient Descent** - Balances batch and stochastic approaches

### 🔬 Problem Types
- **1D Linear Regression** - Fit a line to 1D data
- **2D Linear Regression** - Fit a plane to 2D data

### 📈 Real-time Visualizations
- Loss curves over iterations
- Interactive data scatter plots with fitted functions
- 3D surface plots for 2D problems
- Live metrics and convergence tracking

### ⚙️ Interactive Controls
- Customize sample size and noise
- Adjust learning rate and iterations
- Compare different algorithms
- Reset and experiment instantly

## Project Structure

```
m8_math_foundataion_of_cs/
├── src/
│   ├── __init__.py            
│   ├── gradient_descent.py     # 3 gradient descent algorithms
│   ├── app.py                  # Flask API server
│   └── utils.py                # Data generation utilities
├── static/
│   ├── css/style.css           # Stylesheet
│   ├── js/app.js               # Frontend JavaScript
│   └── html/index.html         # Main page
├── sample/
│   ├── sample_1d_gd.py         # Reference 1D code
│   └── sample_multi_dim_gd.py  # Reference 2D code
├── test_algorithms.py           
├── README.md                    
├── QUICKSTART.md                
└── pyproject.toml               
```

## Installation & Setup

### 1. Install Dependencies

```bash
uv sync
```

### 2. Start the Flask Server

```bash
python -m src.app
```

Server will be available at `http://localhost:3000`

## How to Use

### Step 1: Configure Data
1. Choose **Problem Type** (1D or 2D)
2. Set number of **Samples** (10-500)
3. Adjust **Noise** level with the slider
4. Click **Generate Data**

### Step 2: Select Algorithm
Choose from:
- **Vanilla Gradient Descent** - Standard batch GD
- **Stochastic GD** - Fast but noisy updates
- **Mini-batch GD** - Balanced approach

### Step 3: Tune Hyperparameters
- **Learning Rate (α)** - Controls step size (0.001-0.5)
  - Smaller: Slower but more stable
  - Larger: Faster but may overshoot
- **Max Iterations** - Maximum optimization steps (100-10,000)

### Step 4: Run & Visualize
Click **Run Optimization** and observe:
- Loss decreasing over iterations
- Data points with fitted line/surface
- Final coefficients and metrics

## Algorithm Details

### Vanilla Gradient Descent
```
θ := θ - α · (1/n) · Σ(h(x_i) - y_i) · x_i
```
Updates using entire dataset. Stable and smooth convergence, but slower.

### Stochastic Gradient Descent
```
θ := θ - α · (h(x_i) - y_i) · x_i  [for each sample]
```
Updates using one sample at a time. Fast but noisier, may not converge smoothly.

### Mini-batch Gradient Descent
```
θ := θ - α · (1/batch_size) · Σ(h(x_i) - y_i) · x_i  [per batch]
```
Updates using small batches. Good balance between speed and stability.

## API Endpoints

### `POST /api/generate-data`
Generate synthetic training data.

**Request:**
```json
{
  "problem_type": "1d" | "2d",
  "num_samples": 50,
  "noise_std": 0.5
}
```

**Response:**
```json
{
  "X": [[x1], [x2], ...],
  "y": [y1, y2, ...],
  "problem_type": "1d",
  "num_samples": 50
}
```

### `POST /api/optimize`
Run optimization with specified algorithm and hyperparameters.

**Request:**
```json
{
  "algorithm": "vanilla_gd" | "sgd" | "mini_batch",
  "learning_rate": 0.01,
  "max_iterations": 1000,
  "tolerance": 1e-8,
  "batch_size": 32
}
```

**Response:**
```json
{
  "algorithm": "vanilla_gd",
  "final_coefficients": [3.0, 2.0],
  "history": {
    "loss": [1.5, 1.2, 0.9, ...],
    "coefficients": [[0, 0], [0.1, 0.05], ...]
  },
  "num_iterations": 245,
  "final_loss": 0.0123
}
```

### `POST /api/predict`
Generate predictions for visualization.

**Request:**
```json
{
  "coefficients": [c0, c1, c2, ...]
}
```

### `GET /api/algorithms`
Get information about available algorithms.

## Implementation Notes

### Minimal NumPy Usage
The algorithms are implemented using basic Python loops rather than NumPy magic, making the mathematics transparent and educational:
- Gradients computed element-wise
- No matrix operations (vectorization avoided)
- Clear parameter update steps

### Key Design Decisions
1. **Pure Python Math** - Easy to understand and modify
2. **Modular Architecture** - Optimize, visualize, compare independently
3. **Session-based API** - Data persists across API calls
4. **Real-time Feedback** - Live progress and metrics

## Example Experiments

### Experiment 1: Learning Rate Impact
1. Set problem to **1D**, generate data
2. Run **Vanilla GD** with α = 0.001 (slow convergence)
3. Run again with α = 0.1 (fast convergence)
4. Compare loss curves

### Experiment 2: Algorithm Comparison
1. Generate **1D** data with high noise (1.5)
2. Compare all three algorithms with same hyperparameters
3. Observe convergence smoothness and stability

### Experiment 3: Problem Complexity
1. Try **2D** problems with different noise levels
2. Adjust max iterations to see convergence
3. Visualize the fitted plane in 3D

## Performance Tips

- **Fast Convergence**: Use SGD with lower learning rate
- **Smooth Convergence**: Use Vanilla GD with higher learning rate
- **Balanced**: Use Mini-batch GD (batch_size=32-64)
- **Noisy Data**: Increase max iterations
- **Clean Data**: Can use fewer iterations

## Troubleshooting

### "Connection refused" error
- Make sure Flask server is running: `python -m src.app`
- Check that port 5000 is available

### Optimization doesn't converge
- **Learning rate too high**: Reduce it (0.001-0.01)
- **Learning rate too low**: May need more iterations
- **Try SGD**: Better for noisy data

### 3D plot not showing for 2D problems
- Make sure Plotly.js CDN is accessible
- Check browser console for errors

## References

The algorithms are based on standard optimization techniques:
- **Gradient Descent**: Ruder, S. (2016). An overview of gradient descent optimization algorithms
- **Stochastic Methods**: Bottou, L. (2012). Stochastic gradient descent tricks
- **Mini-batch Methods**: Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization

## Future Enhancements

- [ ] Add polynomial regression
- [ ] Implement momentum-based methods (Momentum, Adam)
- [ ] Add regularization (L1/L2)
- [ ] Support logistic regression
- [ ] Export results as CSV
- [ ] Parameter sweep visualization

## License

This is an educational project for the course: Math Foundations of Computer Science

## Author

Created as a class project to visualize and understand gradient-based optimization algorithms.

---

**Happy optimizing! 🚀**
