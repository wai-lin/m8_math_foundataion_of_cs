const API_URL = 'http://localhost:3000';
let currentData = null;
let chart1D = null;
let chartLoss = null;
let currentProblemType = '1d';

// Update slider values
document.getElementById('noise').addEventListener('input', (e) => {
    document.getElementById('noiseValue').textContent = e.target.value;
});

document.getElementById('lr').addEventListener('input', (e) => {
    document.getElementById('lrValue').textContent = e.target.value;
});

function setProblemType(type) {
    currentProblemType = type;
    
    // Update tab buttons
    document.querySelectorAll('.tab-button').forEach(btn => {
        btn.classList.remove('active');
    });
    document.querySelector(`[data-type="${type}"]`).classList.add('active');
    
    // Update visualization panels
    const is1D = type === '1d';
    document.getElementById('fitContainer').style.display = is1D ? 'block' : 'none';
    document.getElementById('plotContainer').style.display = !is1D ? 'block' : 'none';
    
    // Clear previous data
    clearCharts();
    document.getElementById('metrics').classList.add('hidden');
    document.getElementById('infoBox').style.display = 'none';
}

function showStatus(message, type = 'success') {
    const statusEl = document.getElementById('status');
    statusEl.textContent = message;
    statusEl.className = `status ${type}`;
}

function clearCharts() {
    if (chart1D) chart1D.destroy();
    if (chartLoss) chartLoss.destroy();
    Plotly.purge('2dPlot');
}

async function generateData() {
    try {
        showStatus('⏳ Generating data...', 'loading');
        
        const response = await fetch(`${API_URL}/api/generate-data`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                problem_type: currentProblemType,
                num_samples: parseInt(document.getElementById('numSamples').value),
                noise_std: parseFloat(document.getElementById('noise').value)
            })
        });

        if (!response.ok) throw new Error('Failed to generate data');
        
        currentData = await response.json();
        document.getElementById('optimizeBtn').disabled = false;
        showStatus(`✓ Data generated: ${currentData.num_samples} samples`, 'success');
        
        // Plot the data
        plot1DData();
    } catch (error) {
        showStatus(`✗ Error: ${error.message}`, 'error');
        console.error(error);
    }
}

function plot1DData() {
    if (currentProblemType !== '1d' || !currentData) return;

    const ctx = document.getElementById('1dChart').getContext('2d');
    
    if (chart1D) chart1D.destroy();
    
    chart1D = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Data Points',
                data: currentData.X.map((x, i) => ({
                    x: x[0],
                    y: currentData.y[i]
                })),
                backgroundColor: 'rgba(102, 126, 234, 0.6)',
                borderColor: 'rgba(102, 126, 234, 1)',
                borderWidth: 1,
                pointRadius: 5,
                pointHoverRadius: 7,
                showLine: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { font: { size: 12 } }
                }
            },
            scales: {
                x: { type: 'linear' },
                y: { type: 'linear' }
            }
        }
    });
}

async function runOptimization() {
    try {
        if (!currentData) {
            showStatus('✗ Please generate data first', 'error');
            return;
        }

        showStatus('⏳ Running optimization...', 'loading');
        document.getElementById('optimizeBtn').disabled = true;

        const response = await fetch(`${API_URL}/api/optimize`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                algorithm: document.getElementById('algorithm').value,
                learning_rate: parseFloat(document.getElementById('lr').value),
                max_iterations: parseInt(document.getElementById('maxIter').value),
                batch_size: 32
            })
        });

        if (!response.ok) throw new Error('Optimization failed');
        
        const result = await response.json();
        
        // Display metrics
        document.getElementById('finalLoss').textContent = result.final_loss.toFixed(6);
        document.getElementById('iterations').textContent = result.num_iterations;
        document.getElementById('metrics').classList.remove('hidden');

        // Display coefficients
        let coeffDisplay = '';
        if (currentProblemType === '1d') {
            coeffDisplay = `c₀ (intercept) = ${result.final_coefficients[0].toFixed(4)}<br>`;
            coeffDisplay += `c₁ (slope) = ${result.final_coefficients[1].toFixed(4)}`;
        } else {
            coeffDisplay = `c₀ (intercept) = ${result.final_coefficients[0].toFixed(4)}<br>`;
            coeffDisplay += `c₁ (coef) = ${result.final_coefficients[1].toFixed(4)}<br>`;
            coeffDisplay += `c₂ (coef) = ${result.final_coefficients[2].toFixed(4)}`;
        }
        document.getElementById('coefficientsDisplay').innerHTML = coeffDisplay;
        document.getElementById('infoBox').style.display = 'block';

        // Plot loss
        plotLoss(result.history.loss);

        // Plot fitted function (for 1D)
        if (currentProblemType === '1d') {
            await plotFittedLine(result.final_coefficients);
        } else {
            await plot2DSurface(result.final_coefficients);
        }

        showStatus(`✓ Optimization complete! (${document.getElementById('algorithm').options[document.getElementById('algorithm').selectedIndex].text})`, 'success');
    } catch (error) {
        showStatus(`✗ Error: ${error.message}`, 'error');
        console.error(error);
    } finally {
        document.getElementById('optimizeBtn').disabled = false;
    }
}

function plotLoss(lossHistory) {
    const ctx = document.getElementById('lossChart').getContext('2d');
    
    if (chartLoss) chartLoss.destroy();
    
    chartLoss = new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array.from({length: lossHistory.length}, (_, i) => i),
            datasets: [{
                label: 'Loss',
                data: lossHistory,
                borderColor: 'rgba(102, 126, 234, 1)',
                backgroundColor: 'rgba(102, 126, 234, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4,
                pointRadius: 0,
                pointHoverRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { font: { size: 12 } } }
            },
            scales: {
                y: { beginAtZero: true }
            }
        }
    });
}

async function plotFittedLine(coefficients) {
    try {
        const response = await fetch(`${API_URL}/api/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ coefficients })
        });

        const predData = await response.json();
        
        // Update chart with fitted line
        const ctx = document.getElementById('1dChart');
        chart1D.data.datasets.push({
            label: `Fitted Line (y = ${coefficients[0].toFixed(2)} + ${coefficients[1].toFixed(2)}x)`,
            data: predData.x.map((xi, i) => ({
                x: xi,
                y: predData.y[i]
            })),
            backgroundColor: 'transparent',
            borderColor: 'rgba(220, 53, 69, 1)',
            borderWidth: 3,
            tension: 0.1,
            pointRadius: 0,
            fill: false,
            showLine: true
        });
        chart1D.update();
    } catch (error) {
        console.error('Error plotting fitted line:', error);
    }
}

async function plot2DSurface(coefficients) {
    try {
        const response = await fetch(`${API_URL}/api/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ coefficients })
        });

        const predData = await response.json();
        
        // Extract unique x1 and x2 values
        const x1Vals = [...new Set(predData.surface.map(p => p.x1))].sort((a, b) => a - b);
        const x2Vals = [...new Set(predData.surface.map(p => p.x2))].sort((a, b) => a - b);
        
        // Create Z matrix
        const Z = x1Vals.map(x1 => 
            x2Vals.map(x2 => {
                const point = predData.surface.find(p => p.x1 === x1 && p.x2 === x2);
                return point ? point.y : 0;
            })
        );

        const trace1 = {
            x: x2Vals,
            y: x1Vals,
            z: Z,
            type: 'surface',
            colorscale: 'Viridis'
        };

        const trace2 = {
            x: currentData.X.map(x => x[1]),
            y: currentData.X.map(x => x[0]),
            z: currentData.y,
            mode: 'markers',
            type: 'scatter3d',
            marker: {
                size: 5,
                color: 'red',
                opacity: 0.7
            },
            name: 'Data Points'
        };

        const layout = {
            title: '3D Surface: Fitted Plane vs Data Points',
            scene: {
                xaxis: { title: 'x₂' },
                yaxis: { title: 'x₁' },
                zaxis: { title: 'y' }
            }
        };

        Plotly.newPlot('2dPlot', [trace1, trace2], layout, { responsive: true });
    } catch (error) {
        console.error('Error plotting 2D surface:', error);
    }
}

async function resetSession() {
    try {
        await fetch(`${API_URL}/api/reset`, { method: 'POST' });
        currentData = null;
        clearCharts();
        document.getElementById('optimizeBtn').disabled = true;
        document.getElementById('metrics').classList.add('hidden');
        document.getElementById('infoBox').style.display = 'none';
        document.getElementById('status').className = 'status';
        showStatus('✓ Session reset', 'success');
    } catch (error) {
        showStatus(`✗ Error: ${error.message}`, 'error');
    }
}

// Initialize
window.addEventListener('load', () => {
    setProblemType('1d');  // Set initial display state
    showStatus('Ready: Generate data to start', 'success');
});
