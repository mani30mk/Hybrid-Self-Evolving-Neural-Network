import React, { useRef } from 'react';
import { Upload, FileCheck } from 'lucide-react';

const FileInput = ({ label, name, file, onChange, accept = ".csv" }) => {
    const fileInputRef = useRef(null);

    return (
        <div
            className={`upload-box ${file ? 'has-file' : ''}`}
            onClick={() => fileInputRef.current.click()}
        >
            <input
                type="file"
                ref={fileInputRef}
                onChange={(e) => {
                    if (e.target.files?.[0]) onChange(name, e.target.files[0]);
                }}
                accept={accept}
                style={{ display: 'none' }}
            />
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '10px' }}>
                {file ? <FileCheck size={20} /> : <Upload size={20} />}
                <div>
                    <div style={{ fontWeight: 600 }}>{label}</div>
                    <div style={{ fontSize: '0.8rem', opacity: 0.7 }}>
                        {file ? file.name : `Click to Upload ${accept}`}
                    </div>
                </div>
            </div>
        </div>
    );
};

const DataUpload = ({ files, onFileChange, category }) => {
    const isImageCategory = category && [
        "Image Classification",
        "Image Segmentation",
        "Object Detection",
        "Image Generation (Generator Only)"
    ].includes(category);

    return (
        <div className="section">
            <h3 className="label" style={{ fontSize: '1.2rem', marginBottom: '1rem' }}>Data Upload</h3>

            <div style={{
                background: 'rgba(99, 102, 241, 0.05)',
                border: '1px dashed var(--accent-primary)',
                borderRadius: '8px',
                padding: '1rem',
                marginBottom: '1.5rem',
                fontSize: '0.85rem',
                color: 'var(--text-secondary)'
            }}>
                <strong style={{ color: 'var(--accent-primary)', display: 'block', marginBottom: '0.5rem' }}>
                    Required File Format (Strict):
                </strong>
                {(() => {
                    switch (category) {
                        case "Image Classification":
                            return (
                                <div>
                                    <p>Upload a <strong>ZIP file</strong> containing numeric folders:</p>
                                    <ul style={{ listStyle: 'disc', paddingLeft: '1.5rem', margin: 0 }}>
                                        <li>Structure: <code>0/</code>, <code>1/</code>, <code>2/</code>...</li>
                                        <li>Content: Images inside these class folders.</li>
                                    </ul>
                                </div>
                            );
                        case "Image Segmentation":
                            return (
                                <div>
                                    <p>Upload a <strong>ZIP file</strong> with two specific folders:</p>
                                    <ul style={{ listStyle: 'disc', paddingLeft: '1.5rem', margin: 0 }}>
                                        <li><code>images/</code>: Contains original images.</li>
                                        <li><code>masks/</code>: Contains corresponding segmentation masks.</li>
                                        <li>Filenames must match exactly between folders.</li>
                                    </ul>
                                </div>
                            );
                        case "Image Generation (Generator Only)":
                            return (
                                <div>
                                    <p>Upload a <strong>ZIP file</strong> in one of two formats:</p>
                                    <ul style={{ listStyle: 'disc', paddingLeft: '1.5rem', margin: 0 }}>
                                        <li><strong>Mode 1 (Text-to-Image):</strong> Contains <code>text.txt</code> and an <code>images/</code> folder.</li>
                                        <li><strong>Mode 2 (Image-to-Image):</strong> Contains an <code>input/</code> folder and an <code>output/</code> folder.</li>
                                    </ul>
                                </div>
                            );
                        case "Natural Language Processing":
                            return (
                                <div>
                                    <p>Upload a <strong>CSV file</strong>:</p>
                                    <ul style={{ listStyle: 'disc', paddingLeft: '1.5rem', margin: 0 }}>
                                        <li>Must contain a column named <code>X</code> (Text Input).</li>
                                        <li>Optional: <code>y</code> column for labels (if training).</li>
                                    </ul>
                                </div>
                            );
                        case "Time Series / Audio":
                            return (
                                <div>
                                    <p>Upload a <strong>CSV file</strong>:</p>
                                    <ul style={{ listStyle: 'disc', paddingLeft: '1.5rem', margin: 0 }}>
                                        <li>Must contain a column named <code>y</code> (Target).</li>
                                        <li>All other columns are treated as numeric features.</li>
                                    </ul>
                                </div>
                            );
                        default:
                            return (
                                <div>
                                    <p>Select a category to see file requirements.</p>
                                </div>
                            );
                    }
                })()}
            </div>

            <FileInput
                label={isImageCategory ? "Upload Dataset (.zip)" : "Upload Dataset (.csv)"}
                name="dataset"
                file={files.dataset}
                onChange={onFileChange}
                accept={isImageCategory ? ".zip" : ".csv,.zip"}
            />
        </div>
    );
};

export default DataUpload;
