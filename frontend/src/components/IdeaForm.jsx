import React, { useState } from 'react';
import { Sparkles } from 'lucide-react';

const IdeaForm = ({ onPredictCategory, isLoading }) => {
    const [prompt, setPrompt] = useState('');
    const [debouncedPrompt, setDebouncedPrompt] = useState('');

    // Debounce Logic
    React.useEffect(() => {
        const handler = setTimeout(() => {
            setDebouncedPrompt(prompt);
        }, 800); // Wait 800ms after typing stops

        return () => {
            clearTimeout(handler);
        };
    }, [prompt]);

    // Trigger Prediction when debounced value changes
    React.useEffect(() => {
        if (debouncedPrompt.trim().length > 10) { // Min length to avoid junk calls
            onPredictCategory(debouncedPrompt);
        }
    }, [debouncedPrompt]);

    return (
        <div className="section">
            <label className="label">
                Describe your Neural Network Idea
            </label>
            <div style={{ position: 'relative' }}>
                <textarea
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    placeholder="e.g. Create a model to classify dogs and cats..."
                    className="input-field"
                    style={{ minHeight: '120px', resize: 'none' }}
                />
                <div style={{ position: 'absolute', bottom: '15px', right: '15px' }}>
                    {isLoading && <Sparkles size={20} className="spin" style={{ color: 'var(--accent-primary)' }} />}
                </div>
            </div>
            <style>{`
        .spin { animation: spin 1s linear infinite; }
        @keyframes spin { 100% { transform: rotate(360deg); } }
      `}</style>
        </div>
    );
};

export default IdeaForm;
