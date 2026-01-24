import React, { useState } from 'react';
import axios from 'axios';
import { BrainCircuit, Loader2 } from 'lucide-react';

import IdeaForm from './components/IdeaForm';
import CategorySelect from './components/CategorySelect';
import ModelParams from './components/ModelParams';
import DataUpload from './components/DataUpload';
import ResultsDisplay from './components/ResultsDisplay';
import CorporateLoader from './components/CorporateLoader';

import './App.css';

const API_BASE = 'http://localhost:5000/api';

function App() {
  const [loading, setLoading] = useState(false);
  const [step, setStep] = useState(1);

  const [formData, setFormData] = useState({
    prompt: '',
    category: '',
    input_shape: '',
    num_classes: ''
  });

  const [files, setFiles] = useState({
    dataset: null
  });

  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const handlePredictCategory = React.useCallback(async (promptText) => {
    setLoading(true);
    setFormData(prev => ({ ...prev, prompt: promptText }));
    try {
      // JSON payload as per backend update
      const res = await axios.post(`${API_BASE}/predict-category`, { prompt: promptText });
      if (res.data.category) {
        setFormData(prev => ({ ...prev, category: res.data.category }));
      }
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  }, []);

  const handleFileChange = (name, file) => {
    setFiles(prev => ({ ...prev, [name]: file }));
  };

  const handleParamChange = (name, value) => {
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async () => {
    setError(null);

    // Validation
    if (!formData.prompt || !formData.category) {
      setError("Please describe your idea and select a category");
      return;
    }

    if (!files.dataset) {
      setError("Please upload the Dataset file");
      return;
    }

    setLoading(true);
    setStep(2);

    try {
      const data = new FormData();
      data.append('prompt', formData.prompt);
      data.append('category', formData.category);
      data.append('input_shape', formData.input_shape);
      data.append('num_classes', formData.num_classes);
      data.append('dataset', files.dataset);

      const res = await axios.post(`${API_BASE}/predict-architecture`, data);
      setResults(res.data);
      setStep(3);
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.error || "An error occurred during training");
      setStep(1);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <div style={{ textAlign: 'center', marginBottom: '3rem' }}>
        <div style={{
          display: 'inline-flex',
          padding: '1rem',
          background: 'rgba(59, 130, 246, 0.1)', // Blue tint
          borderRadius: '20px',
          color: 'var(--accent-primary)',
          marginBottom: '1rem'
        }}>
          <BrainCircuit size={40} />
        </div>
        <h1 className="title">Self-Evolving Neural Network</h1>
        <p className="subtitle">
          Describe your problem, upload your data, and let the AI evolve the perfect architecture.
        </p>
      </div>

      <div className="card fade-in">
        {error && (
          <div style={{
            background: 'rgba(239, 68, 68, 0.1)',
            color: 'var(--error)',
            padding: '1rem',
            borderRadius: 'var(--radius-md)',
            marginBottom: '1rem',
            border: '1px solid rgba(239, 68, 68, 0.2)'
          }}>
            {error}
          </div>
        )}

        {step === 1 && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
            <IdeaForm
              onPredictCategory={handlePredictCategory}
              isLoading={loading}
            />

            <div style={{ height: '1px', background: 'var(--border-color)' }} />

            <div style={{ display: 'grid', gridTemplateColumns: 'minmax(250px, 1fr) 1fr', gap: '2rem' }}>
              <CategorySelect
                selectedCategory={formData.category}
                onCategoryChange={(val) => handleParamChange('category', val)}
              />
              <ModelParams
                inputShape={formData.input_shape}
                numClasses={formData.num_classes}
                onChange={handleParamChange}
              />
            </div>

            <DataUpload
              files={files}
              onFileChange={handleFileChange}
              category={formData.category}
            />

            <button
              onClick={handleSubmit}
              disabled={loading}
              className="btn btn-primary"
              style={{ padding: '1rem', fontSize: '1.1rem', marginTop: '1rem' }}
            >
              Start Evolution
            </button>
          </div>
        )}

        {step === 2 && (
          <CorporateLoader />
        )}

        {step === 3 && (
          <div>
            <ResultsDisplay results={results} />
            <button
              onClick={() => { setStep(1); setResults(null); }}
              className="btn"
              style={{ width: '100%', marginTop: '1rem', background: 'rgba(255,255,255,0.05)' }}
            >
              Start New Experiment
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
