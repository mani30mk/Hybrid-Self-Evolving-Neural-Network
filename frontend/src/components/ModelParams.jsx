import React from 'react';
import { Settings } from 'lucide-react';

const ModelParams = ({ inputShape, numClasses, onChange }) => {
    return (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
            <div className="section">
                <label className="label" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Settings size={16} />
                    Input Shape (e.g., 28,28,1)
                </label>
                <input
                    type="text"
                    value={inputShape}
                    onChange={(e) => onChange('input_shape', e.target.value)}
                    placeholder="224,224,3 (No parentheses)"
                    className="input-field"
                />
            </div>
            <div className="section">
                <label className="label">
                    Number of Classes
                </label>
                <input
                    type="number"
                    value={numClasses}
                    onChange={(e) => onChange('num_classes', e.target.value)}
                    placeholder="e.g. 10"
                    className="input-field"
                />
            </div>
        </div>
    );
};

export default ModelParams;
