import React from 'react';
import { Layers } from 'lucide-react';

const CATEGORIES = [
    "Image Classification",
    "Image Segmentation",
    "Image Generation (Generator Only)",
    "Natural Language Processing",
    "Time Series / Audio",
    "Object Detection"
];

const CategorySelect = ({ selectedCategory, onCategoryChange }) => {
    return (
        <div className="section">
            <label className="label" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Layers size={16} />
                Problem Category
            </label>
            <select
                value={selectedCategory}
                onChange={(e) => onCategoryChange(e.target.value)}
                className="input-field"
                style={{ cursor: 'pointer', appearance: 'none' }}
            >
                <option value="" disabled>Select a category...</option>
                {Array.isArray(CATEGORIES) && CATEGORIES.map((cat) => (
                    <option key={cat} value={cat} style={{ background: 'var(--bg-card)', color: 'var(--text-primary)' }}>
                        {cat}
                    </option>
                ))}
            </select>
        </div>
    );
};

export default CategorySelect;
