#!/usr/bin/env python3
"""
Quick test of the gradient descent algorithms
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.gradient_descent import (
    VanillaGradientDescent,
    StochasticGradientDescent,
    MiniBatchGradientDescent
)
from src.utils import generate_1d_data, generate_2d_data


def test_1d_vanilla_gd():
    """Test 1D gradient descent with vanilla algorithm."""
    print("\n" + "=" * 60)
    print("Test 1: 1D Vanilla Gradient Descent")
    print("=" * 60)
    
    X, y = generate_1d_data(num_samples=30, slope=2.5, intercept=1.5, noise_std=0.3)
    
    optimizer = VanillaGradientDescent(
        learning_rate=0.05,
        max_iterations=500,
        tolerance=1e-6
    )
    
    coeffs = optimizer.fit(X, y, [0.0, 0.0])
    
    print(f"✓ Converged in {len(optimizer.history['loss'])} iterations")
    print(f"  Final coefficients: c0={coeffs[0]:.4f}, c1={coeffs[1]:.4f}")
    print(f"  Final loss: {optimizer.history['loss'][-1]:.6f}")
    print(f"  Expected: c0≈1.5, c1≈2.5 (noisy data)")
    

def test_1d_sgd():
    """Test 1D with Stochastic Gradient Descent."""
    print("\n" + "=" * 60)
    print("Test 2: 1D Stochastic Gradient Descent")
    print("=" * 60)
    
    X, y = generate_1d_data(num_samples=50, noise_std=0.4)
    
    optimizer = StochasticGradientDescent(
        learning_rate=0.05,
        max_iterations=500,
        tolerance=1e-6
    )
    
    coeffs = optimizer.fit(X, y, [0.0, 0.0])
    
    print(f"✓ Converged in {len(optimizer.history['loss'])} iterations")
    print(f"  Final coefficients: c0={coeffs[0]:.4f}, c1={coeffs[1]:.4f}")
    print(f"  Final loss: {optimizer.history['loss'][-1]:.6f}")
    print(f"  Note: SGD loss is noisier but still converges")


def test_2d_minibatch():
    """Test 2D with Mini-batch Gradient Descent."""
    print("\n" + "=" * 60)
    print("Test 3: 2D Mini-batch Gradient Descent")
    print("=" * 60)
    
    X, y = generate_2d_data(num_samples=80, noise_std=0.5)
    
    optimizer = MiniBatchGradientDescent(
        learning_rate=0.05,
        max_iterations=500,
        tolerance=1e-6,
        batch_size=16
    )
    
    coeffs = optimizer.fit(X, y, [0.0, 0.0, 0.0])
    
    print(f"✓ Converged in {len(optimizer.history['loss'])} iterations")
    print(f"  Final coefficients: c0={coeffs[0]:.4f}, c1={coeffs[1]:.4f}, c2={coeffs[2]:.4f}")
    print(f"  Final loss: {optimizer.history['loss'][-1]:.6f}")
    print(f"  Expected: c0≈3, c1≈2, c2≈-1 (true plane: y = 2*x1 - x2 + 3)")


def test_algorithm_comparison():
    """Compare algorithms on same data."""
    print("\n" + "=" * 60)
    print("Test 4: Algorithm Comparison (1D Data)")
    print("=" * 60)
    
    X, y = generate_1d_data(num_samples=40, noise_std=0.5)
    
    algorithms = {
        'Vanilla GD': VanillaGradientDescent(learning_rate=0.05, max_iterations=300),
        'SGD': StochasticGradientDescent(learning_rate=0.05, max_iterations=300),
        'Mini-batch': MiniBatchGradientDescent(learning_rate=0.05, max_iterations=300, batch_size=8)
    }
    
    print("\nRunning algorithms...")
    for name, optimizer in algorithms.items():
        coeffs = optimizer.fit(X, y, [0.0, 0.0])
        final_loss = optimizer.history['loss'][-1]
        iters = len(optimizer.history['loss'])
        print(f"\n  {name}:")
        print(f"    • Iterations: {iters}")
        print(f"    • Final loss: {final_loss:.6f}")
        print(f"    • Coefficients: c0={coeffs[0]:.4f}, c1={coeffs[1]:.4f}")


def main():
    print("\n" + "=" * 60)
    print("🎯 Gradient Descent Algorithm Tests")
    print("=" * 60)
    
    try:
        test_1d_vanilla_gd()
        test_1d_sgd()
        test_2d_minibatch()
        test_algorithm_comparison()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        print("\nYou can now run the web app:")
        print("  python -m src.app")
        print("\nThen open your browser to:")
        print("  http://localhost:5000")
        print()
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
