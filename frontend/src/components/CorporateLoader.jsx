import React, { useState, useEffect } from 'react';
import { BrainCircuit } from 'lucide-react';

const CorporateLoader = () => {
  const [progress, setProgress] = useState(0);
  const [msgIndex, setMsgIndex] = useState(0);

  const messages = [
    "Analyzing Dimensionality...",
    "Constructing Search Space...",
    "Evolving Neural Topology...",
    "Optimizing Hyperparameters...",
    "Training Candidate Models...",
    "Validating Accuracy Metrics...",
    "Finalizing Architecture..."
  ];

  useEffect(() => {
    // Simulate progress
    const timer = setInterval(() => {
      setProgress((oldProgress) => {
        if (oldProgress >= 90) return 90; // Stall at 90% until done
        const diff = Math.random() * 5;
        return Math.min(oldProgress + diff, 90);
      });
    }, 500);

    // Rotate messages
    const msgTimer = setInterval(() => {
        setMsgIndex(prev => (prev + 1) % messages.length);
    }, 2500);

    return () => {
      clearInterval(timer);
      clearInterval(msgTimer);
    };
  }, []);

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '4rem 2rem',
      background: 'rgba(255, 255, 255, 0.01)', // Very subtle
      borderRadius: '24px',
      border: '1px solid var(--border-color)',
      maxWidth: '600px',
      width: '100%',
      margin: '0 auto',
      position: 'relative',
      overflow: 'hidden'
    }}>

      {/* Background Glow (Blue) */}
      <div style={{
        position: 'absolute',
        top: '50%',
        left: '50%',
        transform: 'translate(-50%, -50%)',
        width: '300px',
        height: '300px',
        background: 'radial-gradient(circle, rgba(59, 130, 246, 0.1) 0%, transparent 70%)',
        zIndex: 0,
        pointerEvents: 'none'
      }} />

      {/* Animated Logo Container */}
      <div style={{
        position: 'relative',
        zIndex: 1,
        marginBottom: '2.5rem'
      }}>
        {/* Pulse Rings */}
        <div className="pulse-ring" style={{ animationDelay: '0s' }} />
        <div className="pulse-ring" style={{ animationDelay: '1s' }} />

        {/* Logo */}
        <div style={{
           background: 'rgba(59, 130, 246, 0.05)',
           padding: '2rem',
           borderRadius: '50%',
           border: '1px solid rgba(59, 130, 246, 0.2)',
           position: 'relative',
           zIndex: 2,
           boxShadow: '0 0 30px rgba(59, 130, 246, 0.15)'
        }}>
            <BrainCircuit
                size={80}
                style={{
                    color: 'var(--accent-primary)',
                    filter: 'drop-shadow(0 0 10px rgba(59, 130, 246, 0.5))',
                    animation: 'breathe 3s ease-in-out infinite'
                }}
            />
        </div>
      </div>

      {/* Status Text */}
      <div style={{ zIndex: 1, textAlign: 'center', width: '100%', maxWidth: '400px' }}>
        <h3 style={{
            fontSize: '1.8rem',
            fontWeight: 700,
            marginBottom: '0.5rem',
            color: '#fff',
            letterSpacing: '-0.5px'
        }}>
            AI Model Evolution
        </h3>

        <p style={{
            color: 'var(--text-secondary)',
            fontSize: '1rem',
            minHeight: '1.5em',
            marginBottom: '1.5rem',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '8px'
        }}>
            <span style={{
                display: 'inline-block',
                width: '8px',
                height: '8px',
                borderRadius: '50%',
                background: 'var(--accent-primary)',  /* Changed to Blue */
                animation: 'blink 1s infinite'
            }} />
            {messages[msgIndex]}
        </p>

        {/* Progress Bar Container */}
        <div style={{
            width: '100%',
            height: '4px',
            background: 'var(--border-color)',
            borderRadius: '4px',
            overflow: 'hidden',
            position: 'relative'
        }}>
            {/* Progress Bar Fill */}
            <div style={{
                position: 'absolute',
                left: 0,
                top: 0,
                height: '100%',
                width: `${progress}%`,
                background: 'var(--accent-primary)',
                transition: 'width 0.5s ease-out',
                boxShadow: '0 0 10px var(--accent-primary)'
            }} />
        </div>

        <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            marginTop: '0.5rem',
            fontSize: '0.8rem',
            color: 'var(--text-muted)'
        }}>
            <span>System Active</span>
            <span>{Math.round(progress)}%</span>
        </div>
      </div>

      <style>{`
        .pulse-ring {
          position: absolute;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
          width: 100%;
          height: 100%;
          border-radius: 50%;
          border: 1px solid var(--accent-primary);
          opacity: 0;
          animation: pulse-ring 3s cubic-bezier(0.215, 0.61, 0.355, 1) infinite;
        }

        @keyframes pulse-ring {
          0% { transform: translate(-50%, -50%) scale(0.8); opacity: 0.8; border-width: 2px; }
          100% { transform: translate(-50%, -50%) scale(2.5); opacity: 0; border-width: 0px; }
        }

        @keyframes breathe {
          0%, 100% { transform: scale(1); filter: drop-shadow(0 0 10px rgba(59, 130, 246, 0.5)); }
          50% { transform: scale(1.05); filter: drop-shadow(0 0 20px rgba(59, 130, 246, 0.7)); }
        }

        @keyframes blink {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }
      `}</style>
    </div>
  );
};

export default CorporateLoader;
