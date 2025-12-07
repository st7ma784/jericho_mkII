/**
 * Jericho Mk II Real-time Visualization
 * WebGL-accelerated rendering of plasma simulation data
 */

// Global state
let ws = null;
let currentData = {
    fields: null,
    particles: null,
    diagnostics: [],
    step: 0
};

let renderSettings = {
    fieldVectors: false,
    particleTrails: false,
    streamlines: false,
    chargeContours: false,
    fieldType: 'Bz',
    particleColor: 'type',
    phaseColor: 'density',
    currentComponent: 'magnitude',
    flowOverlay: 'speed',
    pressureType: 'thermal'
};

// Canvas contexts
const canvases = {
    field: document.getElementById('field-canvas'),
    particle: document.getElementById('particle-canvas'),
    diagnostics: document.getElementById('diagnostics-canvas'),
    phase: document.getElementById('phase-canvas'),
    current: document.getElementById('current-canvas'),
    flow: document.getElementById('flow-canvas'),
    charge: document.getElementById('charge-canvas'),
    pressure: document.getElementById('pressure-canvas'),
    boundary: document.getElementById('boundary-canvas')
};

const contexts = {};
Object.keys(canvases).forEach(key => {
    contexts[key] = canvases[key].getContext('2d');
});

// Initialize WebSocket connection
function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    
    ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
        console.log('WebSocket connected');
        showConnectionStatus('Connected', true);
        
        // Request initial data
        ws.send(JSON.stringify({
            command: 'set_output_dir',
            path: './output'
        }));
    };
    
    ws.onmessage = (event) => {
        const message = JSON.parse(event.data);
        handleMessage(message);
    };
    
    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        showConnectionStatus('Connection Error', false);
    };
    
    ws.onclose = () => {
        console.log('WebSocket closed, reconnecting...');
        showConnectionStatus('Reconnecting...', false);
        setTimeout(connectWebSocket, 2000);
    };
}

function showConnectionStatus(message, isConnected) {
    const statusDiv = document.getElementById('connection-status');
    statusDiv.textContent = message;
    statusDiv.style.display = 'block';
    
    if (isConnected) {
        statusDiv.classList.add('connected');
        setTimeout(() => {
            statusDiv.style.display = 'none';
        }, 2000);
    } else {
        statusDiv.classList.remove('connected');
    }
}

function handleMessage(message) {
    switch (message.type) {
        case 'fields':
            currentData.fields = message.data;
            renderFields();
            break;
            
        case 'particles':
            currentData.particles = message.data;
            renderParticles();
            renderPhaseSpace();
            updateParticleInfo();
            break;
            
        case 'diagnostics':
            currentData.diagnostics = message.data;
            renderDiagnostics();
            break;
            
        case 'state':
            updateSimulationStatus(message.data);
            break;
            
        case 'error':
            console.error('Server error:', message.message);
            break;
    }
}

function updateSimulationStatus(state) {
    const indicator = document.getElementById('sim-status');
    const statusText = document.getElementById('sim-status-text');
    const stepElem = document.getElementById('current-step');
    const timeElem = document.getElementById('current-time');
    
    if (state.running) {
        indicator.classList.add('running');
        statusText.textContent = 'Running';
    } else {
        indicator.classList.remove('running');
        statusText.textContent = 'Idle';
    }
    
    stepElem.textContent = state.current_step;
    timeElem.textContent = (state.time || 0).toFixed(2);
    currentData.step = state.current_step;
}

// Field rendering
function renderFields() {
    if (!currentData.fields) return;
    
    const canvas = canvases.field;
    const ctx = contexts.field;
    
    // Resize canvas to match container
    resizeCanvas(canvas);
    
    const data = currentData.fields[renderSettings.fieldType];
    if (!data) return;
    
    const [ny, nx] = currentData.fields.shape;
    const cellWidth = canvas.width / nx;
    const cellHeight = canvas.height / ny;
    
    // Find min/max for color scaling
    let min = Infinity, max = -Infinity;
    for (let row of data) {
        for (let val of row) {
            if (val < min) min = val;
            if (val > max) max = val;
        }
    }
    
    const range = max - min || 1;
    
    // Render field as heatmap
    for (let j = 0; j < ny; j++) {
        for (let i = 0; i < nx; i++) {
            const value = data[j][i];
            const normalized = (value - min) / range;
            const color = getColorForValue(normalized);
            
            ctx.fillStyle = color;
            ctx.fillRect(i * cellWidth, j * cellHeight, cellWidth + 1, cellHeight + 1);
        }
    }
    
    // Draw vectors if enabled
    if (renderSettings.fieldVectors && currentData.fields.Ex && currentData.fields.Ey) {
        drawFieldVectors(ctx, canvas, currentData.fields, nx, ny);
    }
    
    // Update info
    document.getElementById('field-info').textContent = 
        `${renderSettings.fieldType}: [${min.toExponential(2)}, ${max.toExponential(2)}]`;
}

function drawFieldVectors(ctx, canvas, fields, nx, ny) {
    const stride = Math.max(1, Math.floor(nx / 40));
    const cellWidth = canvas.width / nx;
    const cellHeight = canvas.height / ny;
    
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.6)';
    ctx.lineWidth = 1;
    
    for (let j = 0; j < ny; j += stride) {
        for (let i = 0; i < nx; i += stride) {
            const Ex = fields.Ex[j][i];
            const Ey = fields.Ey[j][i];
            const mag = Math.sqrt(Ex * Ex + Ey * Ey);
            
            if (mag < 1e-10) continue;
            
            const x = (i + 0.5) * cellWidth;
            const y = (j + 0.5) * cellHeight;
            const scale = Math.min(cellWidth, cellHeight) * 0.4 / mag;
            const dx = Ex * scale;
            const dy = Ey * scale;
            
            ctx.beginPath();
            ctx.moveTo(x, y);
            ctx.lineTo(x + dx, y + dy);
            ctx.stroke();
            
            // Arrow head
            const angle = Math.atan2(dy, dx);
            const headLength = 3;
            ctx.beginPath();
            ctx.moveTo(x + dx, y + dy);
            ctx.lineTo(
                x + dx - headLength * Math.cos(angle - Math.PI / 6),
                y + dy - headLength * Math.sin(angle - Math.PI / 6)
            );
            ctx.moveTo(x + dx, y + dy);
            ctx.lineTo(
                x + dx - headLength * Math.cos(angle + Math.PI / 6),
                y + dy - headLength * Math.sin(angle + Math.PI / 6)
            );
            ctx.stroke();
        }
    }
}

// Particle rendering
function renderParticles() {
    if (!currentData.particles) return;
    
    const canvas = canvases.particle;
    const ctx = contexts.particle;
    
    resizeCanvas(canvas);
    
    if (!renderSettings.particleTrails) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    } else {
        // Fade previous frame
        ctx.fillStyle = 'rgba(10, 14, 39, 0.1)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
    
    const { x, y, vx, vy, v_magnitude, type } = currentData.particles;
    const n = x.length;
    
    // Find domain bounds
    let xMin = Math.min(...x), xMax = Math.max(...x);
    let yMin = Math.min(...y), yMax = Math.max(...y);
    const xRange = xMax - xMin || 1;
    const yRange = yMax - yMin || 1;
    
    // Color scale for velocity
    let vMin = 0, vMax = Math.max(...v_magnitude);
    
    for (let i = 0; i < n; i++) {
        const px = ((x[i] - xMin) / xRange) * canvas.width;
        const py = ((y[i] - yMin) / yRange) * canvas.height;
        
        // Determine color
        let color;
        if (renderSettings.particleColor === 'type') {
            color = type[i] === 0 ? '#66d9ef' : '#f92672'; // Ions vs electrons
        } else if (renderSettings.particleColor === 'velocity') {
            const normalized = (v_magnitude[i] - vMin) / (vMax - vMin || 1);
            color = getColorForValue(normalized);
        } else if (renderSettings.particleColor === 'vx') {
            const normalized = (vx[i] - Math.min(...vx)) / (Math.max(...vx) - Math.min(...vx) || 1);
            color = getColorForValue(normalized);
        } else if (renderSettings.particleColor === 'vy') {
            const normalized = (vy[i] - Math.min(...vy)) / (Math.max(...vy) - Math.min(...vy) || 1);
            color = getColorForValue(normalized);
        }
        
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(px, py, 1.5, 0, Math.PI * 2);
        ctx.fill();
    }
}

function updateParticleInfo() {
    if (!currentData.particles) return;
    
    const { count, total_count } = currentData.particles;
    document.getElementById('particle-info').textContent = 
        `Particles: ${count.toLocaleString()} (${total_count.toLocaleString()} total)`;
}

// Diagnostics rendering
function renderDiagnostics() {
    if (currentData.diagnostics.length === 0) return;
    
    const canvas = canvases.diagnostics;
    const ctx = contexts.diagnostics;
    
    resizeCanvas(canvas);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const diag = currentData.diagnostics;
    const n = diag.length;
    
    if (n < 2) return;
    
    // Extract time series
    const times = diag.map(d => d.time || d.step || 0);
    const totalEnergy = diag.map(d => d.total_energy || 0);
    const kineticEnergy = diag.map(d => d.kinetic_energy || 0);
    const fieldEnergy = diag.map(d => d.field_energy || 0);
    
    // Find ranges
    const tMin = Math.min(...times);
    const tMax = Math.max(...times);
    const eMin = Math.min(...totalEnergy, ...kineticEnergy, ...fieldEnergy);
    const eMax = Math.max(...totalEnergy, ...kineticEnergy, ...fieldEnergy);
    
    const margin = { left: 50, right: 20, top: 20, bottom: 40 };
    const plotWidth = canvas.width - margin.left - margin.right;
    const plotHeight = canvas.height - margin.top - margin.bottom;
    
    // Draw axes
    ctx.strokeStyle = '#4a5588';
    ctx.lineWidth = 2;
    ctx.strokeRect(margin.left, margin.top, plotWidth, plotHeight);
    
    // Draw grid
    ctx.strokeStyle = '#2d3561';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 5; i++) {
        const y = margin.top + (plotHeight * i) / 5;
        ctx.beginPath();
        ctx.moveTo(margin.left, y);
        ctx.lineTo(margin.left + plotWidth, y);
        ctx.stroke();
    }
    
    // Plot data
    function plotLine(data, color) {
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        
        for (let i = 0; i < n; i++) {
            const x = margin.left + ((times[i] - tMin) / (tMax - tMin || 1)) * plotWidth;
            const y = margin.top + plotHeight - ((data[i] - eMin) / (eMax - eMin || 1)) * plotHeight;
            
            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        ctx.stroke();
    }
    
    plotLine(totalEnergy, '#66d9ef');
    plotLine(kineticEnergy, '#a6e22e');
    plotLine(fieldEnergy, '#f92672');
    
    // Labels
    ctx.fillStyle = '#e0e0e0';
    ctx.font = '11px monospace';
    ctx.fillText(`E: [${eMin.toExponential(1)}, ${eMax.toExponential(1)}]`, margin.left, margin.top - 5);
    ctx.fillText(`t: [${tMin.toFixed(1)}, ${tMax.toFixed(1)}]`, margin.left, canvas.height - 5);
}

// Phase space rendering
function renderPhaseSpace() {
    if (!currentData.particles) return;
    
    const canvas = canvases.phase;
    const ctx = contexts.phase;
    
    resizeCanvas(canvas);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const { vx, vy, v_magnitude } = currentData.particles;
    const n = vx.length;
    
    // Find velocity bounds
    const vxMin = Math.min(...vx), vxMax = Math.max(...vx);
    const vyMin = Math.min(...vy), vyMax = Math.max(...vy);
    const vxRange = vxMax - vxMin || 1;
    const vyRange = vyMax - vyMin || 1;
    
    // Create 2D histogram
    const bins = 64;
    const hist = Array(bins).fill(0).map(() => Array(bins).fill(0));
    
    for (let i = 0; i < n; i++) {
        const ix = Math.floor(((vx[i] - vxMin) / vxRange) * (bins - 1));
        const iy = Math.floor(((vy[i] - vyMin) / vyRange) * (bins - 1));
        hist[iy][ix]++;
    }
    
    // Find max for normalization
    let maxCount = 0;
    for (let row of hist) {
        maxCount = Math.max(maxCount, ...row);
    }
    
    // Draw histogram
    const cellWidth = canvas.width / bins;
    const cellHeight = canvas.height / bins;
    
    for (let j = 0; j < bins; j++) {
        for (let i = 0; i < bins; i++) {
            const value = hist[j][i] / (maxCount || 1);
            if (value > 0) {
                const color = getColorForValue(value);
                ctx.fillStyle = color;
                ctx.fillRect(i * cellWidth, j * cellHeight, cellWidth + 1, cellHeight + 1);
            }
        }
    }
    
    // Draw axes
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
    ctx.lineWidth = 1;
    const centerX = canvas.width * (-vxMin) / vxRange;
    const centerY = canvas.height * (1 + vyMin / vyRange);
    ctx.beginPath();
    ctx.moveTo(centerX, 0);
    ctx.lineTo(centerX, canvas.height);
    ctx.moveTo(0, centerY);
    ctx.lineTo(canvas.width, centerY);
    ctx.stroke();
}

// Utility functions
function resizeCanvas(canvas) {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    
    if (canvas.width !== rect.width * dpr || canvas.height !== rect.height * dpr) {
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
        const ctx = canvas.getContext('2d');
        ctx.scale(dpr, dpr);
    }
}

function getColorForValue(normalized) {
    // Jet colormap
    const r = Math.max(0, Math.min(255, 255 * (1.5 - 4 * Math.abs(normalized - 0.75))));
    const g = Math.max(0, Math.min(255, 255 * (1.5 - 4 * Math.abs(normalized - 0.5))));
    const b = Math.max(0, Math.min(255, 255 * (1.5 - 4 * Math.abs(normalized - 0.25))));
    
    return `rgb(${Math.round(r)}, ${Math.round(g)}, ${Math.round(b)})`;
}

// UI Controls
document.getElementById('field-select').addEventListener('change', (e) => {
    renderSettings.fieldType = e.target.value;
    renderFields();
});

document.getElementById('particle-color').addEventListener('change', (e) => {
    renderSettings.particleColor = e.target.value;
    renderParticles();
});

document.getElementById('phase-color').addEventListener('change', (e) => {
    renderSettings.phaseColor = e.target.value;
    renderPhaseSpace();
});

document.getElementById('current-component').addEventListener('change', (e) => {
    renderSettings.currentComponent = e.target.value;
    renderCurrentDensity();
});

document.getElementById('flow-overlay').addEventListener('change', (e) => {
    renderSettings.flowOverlay = e.target.value;
    renderFlow();
});

document.getElementById('pressure-type').addEventListener('change', (e) => {
    renderSettings.pressureType = e.target.value;
    renderPressure();
});

function toggleFieldVectors() {
    renderSettings.fieldVectors = !renderSettings.fieldVectors;
    renderFields();
}

function toggleParticleTrails() {
    renderSettings.particleTrails = !renderSettings.particleTrails;
}

function toggleStreamlines() {
    renderSettings.streamlines = !renderSettings.streamlines;
    renderFlow();
}

function toggleChargeContours() {
    renderSettings.chargeContours = !renderSettings.chargeContours;
    renderChargeDensity();
}

// Current density rendering
function renderCurrentDensity() {
    if (!currentData.particles) return;
    
    const canvas = canvases.current;
    const ctx = contexts.current;
    
    resizeCanvas(canvas);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Calculate current density from particle velocities
    const { x, y, vx, vy, vz, type } = currentData.particles;
    const n = x.length;
    
    // Create 2D grid for current
    const bins = 128;
    const Jx = Array(bins).fill(0).map(() => Array(bins).fill(0));
    const Jy = Array(bins).fill(0).map(() => Array(bins).fill(0));
    const Jz = Array(bins).fill(0).map(() => Array(bins).fill(0));
    
    // Find bounds
    const xMin = Math.min(...x), xMax = Math.max(...x);
    const yMin = Math.min(...y), yMax = Math.max(...y);
    const xRange = xMax - xMin || 1;
    const yRange = yMax - yMin || 1;
    
    // Accumulate current: J = q * n * v
    for (let i = 0; i < n; i++) {
        const ix = Math.floor(((x[i] - xMin) / xRange) * (bins - 1));
        const iy = Math.floor(((y[i] - yMin) / yRange) * (bins - 1));
        
        if (ix >= 0 && ix < bins && iy >= 0 && iy < bins) {
            const charge = type[i] === 0 ? 1 : -1; // Ion=+1, electron=-1
            Jx[iy][ix] += charge * vx[i];
            Jy[iy][ix] += charge * vy[i];
            Jz[iy][ix] += charge * vz[i];
        }
    }
    
    // Render selected component
    let data;
    if (renderSettings.currentComponent === 'magnitude') {
        data = Jx.map((row, j) => row.map((_, i) => 
            Math.sqrt(Jx[j][i]**2 + Jy[j][i]**2 + Jz[j][i]**2)
        ));
    } else if (renderSettings.currentComponent === 'Jx') {
        data = Jx;
    } else if (renderSettings.currentComponent === 'Jy') {
        data = Jy;
    } else {
        data = Jz;
    }
    
    renderHeatmap(ctx, canvas, data);
    
    // Update info
    const maxJ = Math.max(...data.flat().map(Math.abs));
    document.getElementById('current-info').textContent = 
        `${renderSettings.currentComponent}: Max = ${maxJ.toExponential(2)} A/m²`;
}

// Plasma flow rendering
function renderFlow() {
    if (!currentData.particles) return;
    
    const canvas = canvases.flow;
    const ctx = contexts.flow;
    
    resizeCanvas(canvas);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Calculate bulk flow velocity on grid
    const { x, y, vx, vy } = currentData.particles;
    const n = x.length;
    const bins = 64;
    
    const vxGrid = Array(bins).fill(0).map(() => Array(bins).fill(0));
    const vyGrid = Array(bins).fill(0).map(() => Array(bins).fill(0));
    const counts = Array(bins).fill(0).map(() => Array(bins).fill(0));
    
    const xMin = Math.min(...x), xMax = Math.max(...x);
    const yMin = Math.min(...y), yMax = Math.max(...y);
    const xRange = xMax - xMin || 1;
    const yRange = yMax - yMin || 1;
    
    // Average velocities in each cell
    for (let i = 0; i < n; i++) {
        const ix = Math.floor(((x[i] - xMin) / xRange) * (bins - 1));
        const iy = Math.floor(((y[i] - yMin) / yRange) * (bins - 1));
        
        if (ix >= 0 && ix < bins && iy >= 0 && iy < bins) {
            vxGrid[iy][ix] += vx[i];
            vyGrid[iy][ix] += vy[i];
            counts[iy][ix]++;
        }
    }
    
    // Normalize
    for (let j = 0; j < bins; j++) {
        for (let i = 0; i < bins; i++) {
            if (counts[j][i] > 0) {
                vxGrid[j][i] /= counts[j][i];
                vyGrid[j][i] /= counts[j][i];
            }
        }
    }
    
    // Render background (speed or vorticity)
    if (renderSettings.flowOverlay === 'speed') {
        const speed = vxGrid.map((row, j) => row.map((_, i) => 
            Math.sqrt(vxGrid[j][i]**2 + vyGrid[j][i]**2)
        ));
        renderHeatmap(ctx, canvas, speed);
    } else {
        // Calculate vorticity: ∇ × v
        const vorticity = Array(bins).fill(0).map(() => Array(bins).fill(0));
        for (let j = 1; j < bins - 1; j++) {
            for (let i = 1; i < bins - 1; i++) {
                const dvy_dx = (vyGrid[j][i+1] - vyGrid[j][i-1]) / 2;
                const dvx_dy = (vxGrid[j+1][i] - vxGrid[j-1][i]) / 2;
                vorticity[j][i] = dvy_dx - dvx_dy;
            }
        }
        renderHeatmap(ctx, canvas, vorticity);
    }
    
    // Draw velocity vectors or streamlines
    if (renderSettings.streamlines) {
        drawStreamlines(ctx, canvas, vxGrid, vyGrid);
    } else {
        drawVectorField(ctx, canvas, vxGrid, vyGrid);
    }
    
    document.getElementById('flow-info').textContent = 
        `${renderSettings.flowOverlay}: Bulk plasma motion`;
}

// Charge density rendering
function renderChargeDensity() {
    if (!currentData.particles) return;
    
    const canvas = canvases.charge;
    const ctx = contexts.charge;
    
    resizeCanvas(canvas);
    
    const { x, y, type } = currentData.particles;
    const n = x.length;
    const bins = 128;
    
    const density = Array(bins).fill(0).map(() => Array(bins).fill(0));
    
    const xMin = Math.min(...x), xMax = Math.max(...x);
    const yMin = Math.min(...y), yMax = Math.max(...y);
    const xRange = xMax - xMin || 1;
    const yRange = yMax - yMin || 1;
    
    // Accumulate charge
    for (let i = 0; i < n; i++) {
        const ix = Math.floor(((x[i] - xMin) / xRange) * (bins - 1));
        const iy = Math.floor(((y[i] - yMin) / yRange) * (bins - 1));
        
        if (ix >= 0 && ix < bins && iy >= 0 && iy < bins) {
            density[iy][ix] += type[i] === 0 ? 1 : -1;
        }
    }
    
    renderHeatmap(ctx, canvas, density);
    
    // Draw contours if enabled
    if (renderSettings.chargeContours) {
        drawContours(ctx, canvas, density);
    }
    
    const maxRho = Math.max(...density.flat().map(Math.abs));
    document.getElementById('charge-info').textContent = 
        `ρ: Max = ${maxRho.toExponential(2)} C/m³ | Quasi-neutral`;
}

// Pressure rendering
function renderPressure() {
    if (!currentData.particles) return;
    
    const canvas = canvases.pressure;
    const ctx = contexts.pressure;
    
    resizeCanvas(canvas);
    
    const { x, y, vx, vy, vz } = currentData.particles;
    const n = x.length;
    const bins = 64;
    
    // Calculate temperature/pressure on grid
    const temp = Array(bins).fill(0).map(() => Array(bins).fill(0));
    const vxMean = Array(bins).fill(0).map(() => Array(bins).fill(0));
    const vyMean = Array(bins).fill(0).map(() => Array(bins).fill(0));
    const counts = Array(bins).fill(0).map(() => Array(bins).fill(0));
    
    const xMin = Math.min(...x), xMax = Math.max(...x);
    const yMin = Math.min(...y), yMax = Math.max(...y);
    const xRange = xMax - xMin || 1;
    const yRange = yMax - yMin || 1;
    
    // First pass: mean velocities
    for (let i = 0; i < n; i++) {
        const ix = Math.floor(((x[i] - xMin) / xRange) * (bins - 1));
        const iy = Math.floor(((y[i] - yMin) / yRange) * (bins - 1));
        
        if (ix >= 0 && ix < bins && iy >= 0 && iy < bins) {
            vxMean[iy][ix] += vx[i];
            vyMean[iy][ix] += vy[i];
            counts[iy][ix]++;
        }
    }
    
    for (let j = 0; j < bins; j++) {
        for (let i = 0; i < bins; i++) {
            if (counts[j][i] > 0) {
                vxMean[j][i] /= counts[j][i];
                vyMean[j][i] /= counts[j][i];
            }
        }
    }
    
    // Second pass: thermal velocity (temperature)
    for (let i = 0; i < n; i++) {
        const ix = Math.floor(((x[i] - xMin) / xRange) * (bins - 1));
        const iy = Math.floor(((y[i] - yMin) / yRange) * (bins - 1));
        
        if (ix >= 0 && ix < bins && iy >= 0 && iy < bins) {
            const dvx = vx[i] - vxMean[iy][ix];
            const dvy = vy[i] - vyMean[iy][ix];
            temp[iy][ix] += dvx**2 + dvy**2;
        }
    }
    
    // Normalize to get thermal pressure P = nkT
    for (let j = 0; j < bins; j++) {
        for (let i = 0; i < bins; i++) {
            if (counts[j][i] > 0) {
                temp[j][i] /= counts[j][i];
            }
        }
    }
    
    renderHeatmap(ctx, canvas, temp);
    
    const avgTemp = temp.flat().reduce((a, b) => a + b, 0) / (bins * bins);
    document.getElementById('pressure-info').textContent = 
        `${renderSettings.pressureType}: Avg = ${avgTemp.toExponential(2)} eV`;
}

// Boundary flux rendering
function renderBoundary() {
    const canvas = canvases.boundary;
    const ctx = contexts.boundary;
    
    resizeCanvas(canvas);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    if (!currentData.particles) return;
    
    const w = canvas.width, h = canvas.height;
    const margin = 40;
    
    // Draw simulation domain
    ctx.strokeStyle = '#4a5588';
    ctx.lineWidth = 2;
    ctx.strokeRect(margin, margin, w - 2*margin, h - 2*margin);
    
    // Draw boundary zones
    const zoneWidth = 20;
    
    // Left boundary (inflow example)
    ctx.fillStyle = 'rgba(166, 226, 46, 0.3)'; // Green = inflow
    ctx.fillRect(margin, margin, zoneWidth, h - 2*margin);
    drawArrow(ctx, margin + zoneWidth/2, h/2, margin + zoneWidth, h/2, '#a6e22e');
    ctx.fillStyle = '#a6e22e';
    ctx.font = '12px monospace';
    ctx.fillText('INFLOW', 5, h/2);
    
    // Right boundary (outflow example)
    ctx.fillStyle = 'rgba(249, 38, 114, 0.3)'; // Red = outflow
    ctx.fillRect(w - margin - zoneWidth, margin, zoneWidth, h - 2*margin);
    drawArrow(ctx, w - margin - zoneWidth, h/2, w - margin - zoneWidth/2, h/2, '#f92672');
    ctx.fillStyle = '#f92672';
    ctx.fillText('OUTFLOW', w - 70, h/2);
    
    // Top/bottom (periodic example)
    ctx.fillStyle = 'rgba(102, 217, 239, 0.3)'; // Cyan = periodic
    ctx.fillRect(margin, margin, w - 2*margin, zoneWidth);
    ctx.fillRect(margin, h - margin - zoneWidth, w - 2*margin, zoneWidth);
    
    drawArrow(ctx, w/2 - 30, margin + zoneWidth/2, w/2 - 10, margin + zoneWidth/2, '#66d9ef');
    drawArrow(ctx, w/2 + 10, h - margin - zoneWidth/2, w/2 + 30, h - margin - zoneWidth/2, '#66d9ef');
    
    ctx.fillStyle = '#66d9ef';
    ctx.fillText('PERIODIC', w/2 - 35, 25);
    ctx.fillText('PERIODIC', w/2 - 35, h - 10);
    
    // Count particles near boundaries
    const { x, y } = currentData.particles;
    const xMin = Math.min(...x), xMax = Math.max(...x);
    const yMin = Math.min(...y), yMax = Math.max(...y);
    const threshold = 0.05;
    
    let inflowCount = 0, outflowCount = 0;
    for (let i = 0; i < x.length; i++) {
        const nx = (x[i] - xMin) / (xMax - xMin);
        if (nx < threshold) inflowCount++;
        if (nx > 1 - threshold) outflowCount++;
    }
    
    document.getElementById('boundary-info').innerHTML = 
        `<span style="color: #a6e22e;">↓ Inflow: ${inflowCount}</span> | ` +
        `<span style="color: #f92672;">↑ Outflow: ${outflowCount}</span> | ` +
        `<span style="color: #66d9ef;">↔ Periodic</span>`;
}

// Helper rendering functions
function renderHeatmap(ctx, canvas, data) {
    const bins = data.length;
    const cellWidth = canvas.width / bins;
    const cellHeight = canvas.height / bins;
    
    let min = Infinity, max = -Infinity;
    for (let row of data) {
        for (let val of row) {
            if (val < min) min = val;
            if (val > max) max = val;
        }
    }
    
    const range = max - min || 1;
    
    for (let j = 0; j < bins; j++) {
        for (let i = 0; i < bins; i++) {
            const normalized = (data[j][i] - min) / range;
            const color = getColorForValue(normalized);
            ctx.fillStyle = color;
            ctx.fillRect(i * cellWidth, j * cellHeight, cellWidth + 1, cellHeight + 1);
        }
    }
}

function drawVectorField(ctx, canvas, vx, vy) {
    const bins = vx.length;
    const cellWidth = canvas.width / bins;
    const cellHeight = canvas.height / bins;
    const stride = Math.max(1, Math.floor(bins / 20));
    
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.7)';
    ctx.lineWidth = 1.5;
    
    for (let j = 0; j < bins; j += stride) {
        for (let i = 0; i < bins; i += stride) {
            const mag = Math.sqrt(vx[j][i]**2 + vy[j][i]**2);
            if (mag < 1e-10) continue;
            
            const x = (i + 0.5) * cellWidth;
            const y = (j + 0.5) * cellHeight;
            const scale = Math.min(cellWidth, cellHeight) * 0.4 / mag;
            const dx = vx[j][i] * scale;
            const dy = vy[j][i] * scale;
            
            ctx.beginPath();
            ctx.moveTo(x, y);
            ctx.lineTo(x + dx, y + dy);
            ctx.stroke();
            
            // Arrow head
            const angle = Math.atan2(dy, dx);
            const headLen = 3;
            ctx.beginPath();
            ctx.moveTo(x + dx, y + dy);
            ctx.lineTo(x + dx - headLen * Math.cos(angle - Math.PI/6), 
                       y + dy - headLen * Math.sin(angle - Math.PI/6));
            ctx.moveTo(x + dx, y + dy);
            ctx.lineTo(x + dx - headLen * Math.cos(angle + Math.PI/6),
                       y + dy - headLen * Math.sin(angle + Math.PI/6));
            ctx.stroke();
        }
    }
}

function drawStreamlines(ctx, canvas, vx, vy) {
    const bins = vx.length;
    const nStreams = 30;
    
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
    ctx.lineWidth = 1;
    
    for (let s = 0; s < nStreams; s++) {
        let x = Math.random() * canvas.width;
        let y = Math.random() * canvas.height;
        
        ctx.beginPath();
        ctx.moveTo(x, y);
        
        for (let step = 0; step < 100; step++) {
            const i = Math.floor((x / canvas.width) * (bins - 1));
            const j = Math.floor((y / canvas.height) * (bins - 1));
            
            if (i < 0 || i >= bins || j < 0 || j >= bins) break;
            
            const mag = Math.sqrt(vx[j][i]**2 + vy[j][i]**2);
            if (mag < 1e-10) break;
            
            const dx = (vx[j][i] / mag) * 2;
            const dy = (vy[j][i] / mag) * 2;
            
            x += dx;
            y += dy;
            
            ctx.lineTo(x, y);
        }
        
        ctx.stroke();
    }
}

function drawContours(ctx, canvas, data) {
    // Simple contour drawing at zero level
    const bins = data.length;
    const cellWidth = canvas.width / bins;
    const cellHeight = canvas.height / bins;
    
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
    ctx.lineWidth = 2;
    
    for (let j = 0; j < bins - 1; j++) {
        for (let i = 0; i < bins - 1; i++) {
            const v00 = data[j][i];
            const v10 = data[j][i+1];
            const v01 = data[j+1][i];
            
            // Check for zero crossing
            if ((v00 > 0 && v10 < 0) || (v00 < 0 && v10 > 0)) {
                ctx.beginPath();
                ctx.moveTo(i * cellWidth, j * cellHeight);
                ctx.lineTo((i+1) * cellWidth, j * cellHeight);
                ctx.stroke();
            }
            if ((v00 > 0 && v01 < 0) || (v00 < 0 && v01 > 0)) {
                ctx.beginPath();
                ctx.moveTo(i * cellWidth, j * cellHeight);
                ctx.lineTo(i * cellWidth, (j+1) * cellHeight);
                ctx.stroke();
            }
        }
    }
}

function drawArrow(ctx, x1, y1, x2, y2, color) {
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 2;
    
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
    
    const angle = Math.atan2(y2 - y1, x2 - x1);
    const headLen = 8;
    
    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - headLen * Math.cos(angle - Math.PI/6),
               y2 - headLen * Math.sin(angle - Math.PI/6));
    ctx.lineTo(x2 - headLen * Math.cos(angle + Math.PI/6),
               y2 - headLen * Math.sin(angle + Math.PI/6));
    ctx.closePath();
    ctx.fill();
}

// Animation loop
function animate() {
    // Render all panels if data available
    if (currentData.particles) {
        renderCurrentDensity();
        renderFlow();
        renderChargeDensity();
        renderPressure();
        renderBoundary();
    }
    
    requestAnimationFrame(animate);
}

// Initialize
window.addEventListener('load', () => {
    connectWebSocket();
    animate();
    
    // Handle window resize
    window.addEventListener('resize', () => {
        renderFields();
        renderParticles();
        renderDiagnostics();
        renderPhaseSpace();
        renderCurrentDensity();
        renderFlow();
        renderChargeDensity();
        renderPressure();
        renderBoundary();
    });
});
