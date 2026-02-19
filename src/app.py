"""
Flask API for Gradient Descent Visualization
"""

from src.utils import generate_1d_data, generate_2d_data, get_algorithm_info
from src.gradient_descent import (
    VanillaGradientDescent,
    StochasticGradientDescent,
    MiniBatchGradientDescent
)
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent


app = Flask(
    __name__,
    static_folder=str(PROJECT_ROOT / 'static'),
    static_url_path='/static'
)
CORS(app)

# Store data for session
session_data = {}


@app.route('/api/algorithms', methods=['GET'])
def get_algorithms():
    """Get available algorithms and their info."""
    return jsonify(get_algorithm_info())


@app.route('/api/generate-data', methods=['POST'])
def generate_data():
    """Generate synthetic data for visualization."""
    data = request.json
    problem_type = data.get('problem_type', '1d')  # '1d' or '2d'
    num_samples = data.get('num_samples', 50)
    noise_std = data.get('noise_std', 0.5)

    if problem_type == '1d':
        X, y = generate_1d_data(num_samples=num_samples, noise_std=noise_std)
    elif problem_type == '2d':
        X, y = generate_2d_data(num_samples=num_samples, noise_std=noise_std)
    else:
        return jsonify({'error': 'Invalid problem_type'}), 400

    # Store data in session (keep as NumPy arrays internally)
    session_data['X'] = X
    session_data['y'] = y
    session_data['problem_type'] = problem_type

    # Convert to lists for JSON response
    X_list = X.tolist() if hasattr(X, 'tolist') else X
    y_list = y.tolist() if hasattr(y, 'tolist') else y

    return jsonify({
        'X': X_list,
        'y': y_list,
        'problem_type': problem_type,
        'num_samples': len(y)
    })


@app.route('/api/optimize', methods=['POST'])
def optimize():
    """Run optimization with specified algorithm and hyperparameters."""
    if 'X' not in session_data or 'y' not in session_data:
        return jsonify({'error': 'No data generated. Call /api/generate-data first'}), 400

    data = request.json
    algorithm = data.get('algorithm', 'vanilla_gd')
    learning_rate = data.get('learning_rate', 0.01)
    max_iterations = data.get('max_iterations', 1000)
    tolerance = data.get('tolerance', 1e-8)

    X = session_data['X']
    y = session_data['y']
    problem_type = session_data['problem_type']

    # Initial coefficients based on problem type
    if problem_type == '1d':
        coeffs_init = [0.0, 0.0]
    else:  # 2d
        coeffs_init = [0.0, 0.0, 0.0]

    # Create optimizer
    if algorithm == 'sgd':
        optimizer = StochasticGradientDescent(
            learning_rate=learning_rate,
            max_iterations=max_iterations,
            tolerance=tolerance
        )
    elif algorithm == 'mini_batch':
        batch_size = data.get('batch_size', 32)
        optimizer = MiniBatchGradientDescent(
            learning_rate=learning_rate,
            max_iterations=max_iterations,
            tolerance=tolerance,
            batch_size=batch_size
        )
    else:  # vanilla_gd
        optimizer = VanillaGradientDescent(
            learning_rate=learning_rate,
            max_iterations=max_iterations,
            tolerance=tolerance
        )

    # Run optimization
    final_coeffs = optimizer.fit(X, y, coeffs_init)

    # Convert NumPy arrays to lists for JSON serialization
    final_coeffs_list = final_coeffs.tolist() if hasattr(
        final_coeffs, 'tolist') else list(final_coeffs)
    history_coeffs = [c.tolist() if hasattr(c, 'tolist') else list(c)
                      for c in optimizer.history['coefficients']]

    # Prepare response with history and final result
    return jsonify({
        'algorithm': algorithm,
        'final_coefficients': final_coeffs_list,
        'history': {
            'loss': [float(l) for l in optimizer.history['loss']],
            'coefficients': history_coeffs,
        },
        'num_iterations': len(optimizer.history['loss']),
        'final_loss': float(optimizer.history['loss'][-1]) if optimizer.history['loss'] else 0
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """Generate predictions at given points (for visualization)."""
    if 'X' not in session_data or 'y' not in session_data:
        return jsonify({'error': 'No data generated'}), 400

    data = request.json
    coefficients = data.get('coefficients')
    problem_type = session_data['problem_type']

    if problem_type == '1d':
        # Generate line for visualization
        num_points = 100
        x_min = min(x[0] for x in session_data['X'])
        x_max = max(x[0] for x in session_data['X'])

        line_x = []
        line_y = []
        for i in range(num_points):
            x = x_min + (x_max - x_min) * i / (num_points - 1)
            y = coefficients[0] + coefficients[1] * x
            line_x.append(x)
            line_y.append(y)

        return jsonify({
            'x': line_x,
            'y': line_y,
            'problem_type': '1d'
        })

    else:  # 2d
        # Generate grid for surface visualization
        x1_vals = sorted(set(x[0] for x in session_data['X']))
        x2_vals = sorted(set(x[1] for x in session_data['X']))

        surface = []
        for x1 in x1_vals:
            for x2 in x2_vals:
                y = coefficients[0] + coefficients[1] * \
                    x1 + coefficients[2] * x2
                surface.append({'x1': x1, 'x2': x2, 'y': y})

        return jsonify({
            'surface': surface,
            'problem_type': '2d'
        })


@app.route('/api/reset', methods=['POST'])
def reset():
    """Reset session data."""
    session_data.clear()
    return jsonify({'message': 'Session reset'})


@app.route('/')
def index():
    """Serve the main page."""
    return send_from_directory(str(PROJECT_ROOT / 'static' / 'html'), 'index.html')


if __name__ == '__main__':
    app.run(port=3000)
