import React from 'react';
import { CheckCircle, Download, Activity, Box, List } from 'lucide-react';
import { Line } from 'react-chartjs-2';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
    Filler
} from 'chart.js';

ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
    Filler
);

const ResultsDisplay = ({ results }) => {
    if (!results) return null;

    const { category, architecture, full_architecture, val_accuracy, parameters, download_url, inference, layers } = results;
    const isNLP = category === "Natural Language Processing";

    // Helper to render inference data based on category
    const renderInference = () => {
        if (!inference || !inference.X_val) return null;

        const { X_val, y_val, y_pred } = inference;
        const isInputImage = category.startsWith("Image") || category === "Object Detection";
        const isOutputImage = category === "Image Segmentation" || category === "Image Generation (Generator Only)";

        return (
            <div className="card" style={{ marginBottom: '1.5rem', overflowX: 'auto' }}>
                <h4 style={{ marginBottom: '1rem', fontWeight: 600 }}>Inference Samples (Validation Set)</h4>
                <div style={{ display: 'flex', gap: '1rem', paddingBottom: '0.5rem' }}>
                    {X_val.map((item, i) => (
                        <div key={i} style={{
                            minWidth: isOutputImage ? '200px' : '150px',
                            border: '1px solid var(--border-color)',
                            borderRadius: '8px',
                            padding: '0.5rem',
                            background: 'var(--bg-secondary)',
                            fontSize: '0.8rem'
                        }}>
                            <div style={{ marginBottom: '0.5rem', fontWeight: 'bold', color: 'var(--text-secondary)' }}>Sample {i + 1}</div>

                            {/* Input Visualization */}
                            {isInputImage ? (
                                <img
                                    src={`data:image/png;base64,${item}`}
                                    alt={`Input ${i}`}
                                    style={{ width: '100%', height: '100px', objectFit: 'cover', borderRadius: '4px', marginBottom: '0.5rem', border: '1px solid var(--border-color)' }}
                                />
                            ) : (
                                <div style={{
                                    maxHeight: '80px',
                                    overflowY: 'auto',
                                    background: 'var(--bg-primary)',
                                    padding: '4px',
                                    borderRadius: '4px',
                                    marginBottom: '0.5rem',
                                    fontFamily: 'monospace',
                                    fontSize: '0.7rem'
                                }}>
                                    {Array.isArray(item) ? JSON.stringify(item) : item}
                                </div>
                            )}

                            {/* Prediction vs Truth */}
                            <div style={{ display: 'grid', gridTemplateColumns: isOutputImage ? '1fr 1fr' : 'auto 1fr', gap: '8px', fontSize: '0.75rem' }}>

                                {/* Ground Truth */}
                                <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                                    <span style={{ color: 'var(--text-muted)' }}>True:</span>
                                    {isOutputImage ? (
                                        <img
                                            src={`data:image/png;base64,${y_val[i]}`}
                                            alt={`True ${i}`}
                                            style={{ width: '100%', height: '60px', objectFit: 'cover', borderRadius: '4px' }}
                                        />
                                    ) : (
                                        <span style={{ fontWeight: 600 }}>{JSON.stringify(y_val[i])}</span>
                                    )}
                                </div>

                                {/* Prediction */}
                                <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                                    <span style={{ color: 'var(--text-muted)' }}>Pred:</span>
                                    {isOutputImage ? (
                                        <img
                                            src={`data:image/png;base64,${y_pred[i]}`}
                                            alt={`Pred ${i}`}
                                            style={{ width: '100%', height: '60px', objectFit: 'cover', borderRadius: '4px', border: '2px solid var(--accent-primary)' }}
                                        />
                                    ) : (
                                        <span style={{ fontWeight: 600, color: JSON.stringify(y_val[i]) === JSON.stringify(y_pred[i]) ? 'var(--success)' : 'var(--error)' }}>
                                            {JSON.stringify(y_pred[i])}
                                        </span>
                                    )}
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        );
    };

    return (
        <div className="section fade-in" style={{ marginTop: '2rem' }}>
            {/* Header Status */}
            <div
                style={{
                    background: 'rgba(16, 185, 129, 0.1)',
                    border: '1px solid rgba(16, 185, 129, 0.2)',
                    padding: '1rem',
                    borderRadius: 'var(--radius-md)',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '1rem',
                    marginBottom: '1.5rem'
                }}
            >
                <CheckCircle color="var(--success)" />
                <div>
                    <h3 style={{ color: 'var(--success)', fontWeight: 600 }}>Model Evolution Complete</h3>
                    <p style={{ fontSize: '0.9rem', opacity: 0.8 }}>Your neural network has been successfully trained.</p>
                </div>
            </div>

            {/* Metrics Grid */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '1rem', marginBottom: '1.5rem' }}>
                <div className="card" style={{ padding: '1.5rem', textAlign: 'center' }}>
                    <div style={{ display: 'flex', justifyContent: 'center', gap: '0.5rem', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>
                        <Activity size={16} />
                        <span style={{ fontSize: '0.8rem' }}>Validation Accuracy</span>
                    </div>
                    <div style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>
                        {(val_accuracy * 100).toFixed(2)}%
                    </div>
                </div>

                <div className="card" style={{ padding: '1.5rem', textAlign: 'center' }}>
                    <div style={{ display: 'flex', justifyContent: 'center', gap: '0.5rem', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>
                        <Box size={16} />
                        <span style={{ fontSize: '0.8rem' }}>Parameters</span>
                    </div>
                    <div style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>
                        {parameters ? parameters.toLocaleString() : 'N/A'}
                    </div>
                </div>

                <div className="card" style={{ padding: '1.5rem', textAlign: 'center' }}>
                    <div style={{ display: 'flex', justifyContent: 'center', gap: '0.5rem', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>
                        <List size={16} />
                        <span style={{ fontSize: '0.8rem' }}>Layers</span>
                    </div>
                    <div style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>
                        {architecture ? architecture.length : 0}
                    </div>
                </div>
            </div>

            {/* Training Graph */}
            {results.val_accuracies && results.val_accuracies.length > 0 && (
                <div className="card" style={{ marginBottom: '1.5rem' }}>
                    <h4 style={{ marginBottom: '1rem', fontWeight: 600 }}>Training Performance</h4>
                    <div style={{ height: '300px' }}>
                        <Line
                            data={{
                                labels: results.epochs ? results.epochs : results.val_accuracies.map((_, i) => i + 1),
                                datasets: [{
                                    label: 'Validation Accuracy',
                                    data: results.val_accuracies,
                                    borderColor: '#10b981',
                                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                                    tension: 0.4,
                                    fill: true,
                                    pointBackgroundColor: '#10b981'
                                }]
                            }}
                            options={{
                                responsive: true,
                                maintainAspectRatio: false,
                                plugins: {
                                    legend: { display: false },
                                    tooltip: {
                                        mode: 'index',
                                        intersect: false,
                                        backgroundColor: 'rgba(0,0,0,0.8)',
                                        titleColor: '#fff',
                                        bodyColor: '#fff'
                                    }
                                },
                                scales: {
                                    y: {
                                        beginAtZero: true,
                                        max: 1.0,
                                        grid: { color: 'rgba(255,255,255,0.05)' },
                                        ticks: { color: 'rgba(255,255,255,0.5)' }
                                    },
                                    x: {
                                        grid: { display: false },
                                        ticks: { color: 'rgba(255,255,255,0.5)' }
                                    }
                                }
                            }}
                        />
                    </div>
                </div>
            )}

            {/* Inference Section */}
            {renderInference()}

            {/* Detailed Architecture */}
            <div className="card" style={{ marginBottom: '1.5rem' }}>
                <h4 style={{ marginBottom: '1rem', fontWeight: 600 }}>Architecture Stack</h4>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                    {layers && full_architecture ? (
                        layers.map((layerName, index) => {
                            // Direct 1:1 mapping since both come from model.layers
                            const layer = full_architecture[index] || {};

                            // Clean up name if it's generic
                            const cleanName = layerName || (layer.class_name || 'Layer');

                            return (
                                <div key={index} style={{
                                    background: 'var(--bg-secondary)',
                                    padding: '0.75rem',
                                    borderRadius: 'var(--radius-sm)',
                                    fontSize: '0.9rem',
                                    border: '1px solid var(--border-color)',
                                    display: 'flex',
                                    justifyContent: 'space-between',
                                    alignItems: 'center'
                                }}>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                                        <span style={{
                                            color: 'var(--text-muted)',
                                            fontSize: '0.8rem',
                                            minWidth: '20px',
                                            height: '20px',
                                            borderRadius: '50%',
                                            background: 'rgba(255,255,255,0.1)',
                                            display: 'flex',
                                            alignItems: 'center',
                                            justifyContent: 'center'
                                        }}>{index + 1}</span>
                                        <span style={{ fontWeight: 600 }}>{cleanName}</span>
                                    </div>
                                    <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', fontFamily: 'monospace' }}>
                                        {layer.units && `Units: ${layer.units} `}
                                        {layer.filters && `Filters: ${layer.filters} `}
                                        {layer.kernel_size && `Kernel: ${JSON.stringify(layer.kernel_size)} `}
                                        {layer.activation && `Act: ${layer.activation} `}
                                        {layer.pool_size && `Pool: ${JSON.stringify(layer.pool_size)} `}
                                        {layer.rate && `Rate: ${layer.rate} `}
                                        {layer.input_dim && `In: ${layer.input_dim} `}
                                        {layer.output_dim && `Out: ${layer.output_dim} `}
                                    </div>
                                </div>
                            );
                        })
                    ) : (
                        // Fallback to simple list
                        results.architecture && results.architecture.map((layer, index) => (
                            <div key={index} style={{ padding: '0.5rem', background: 'var(--bg-secondary)', borderRadius: '4px' }}>
                                {index + 1}. {layer}
                            </div>
                        ))
                    )}
                </div>
            </div>

            {/* Action Buttons */}
            <div style={{ display: 'flex', gap: '1rem', flexDirection: 'column' }}>
                {download_url && (
                    <a
                        href={`http://localhost:5000${download_url}`}
                        download
                        className="btn btn-primary"
                        style={{ width: '100%', textDecoration: 'none', justifyContent: 'center' }}
                    >
                        <Download size={20} />
                        Download Trained Model (.keras)
                    </a>
                )}

                {isNLP && (
                    <a
                        href={`http://localhost:5000${download_url.replace('download-model', 'download-tokenizer').replace('final_model.keras', 'tokenizer.json')}`}
                        download
                        className="btn"
                        style={{ width: '100%', textDecoration: 'none', justifyContent: 'center', background: 'rgba(99, 102, 241, 0.1)', color: 'var(--accent-primary)', border: '1px solid var(--accent-primary)' }}
                    >
                        <Download size={20} />
                        Download Tokenizer (.json)
                    </a>
                )}
            </div>
        </div>
    );
};

export default ResultsDisplay;
