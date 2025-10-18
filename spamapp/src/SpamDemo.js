import './App.css';
import { useEffect, useState } from 'react';

// Backend base URL
const API_BASE = 'http://localhost:8000';

function SpamDemo() {
    const [health, setHealth] = useState(null);
    const [modelInfo, setModelInfo] = useState(null);
    const [text, setText] = useState('');
    const [prediction, setPrediction] = useState(null);
    const [batchResult, setBatchResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    // Basic health + model info fetch on mount
    useEffect(() => {
        const fetchHealthAndInfo = async () => {
            try {
                const healthRes = await fetch(`${API_BASE}/health`);
                if (healthRes.ok) {
                    setHealth(await healthRes.json());
                }
                const modelRes = await fetch(`${API_BASE}/api/v1/model/info`);
                if (modelRes.ok) {
                    setModelInfo(await modelRes.json());
                }
            } catch (e) {
                // swallow network errors for demo
                setError('Backend not reachable. Start FastAPI server.');
            }
        };
        fetchHealthAndInfo();
    }, []);

    const handlePredict = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError(null);
        setPrediction(null);
        try {
            const res = await fetch(`${API_BASE}/api/v1/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text }),
            });
            if (!res.ok) {
                const detail = await res.json();
                throw new Error(detail.detail || 'Prediction failed');
            }
            setPrediction(await res.json());
        } catch (e) {
            setError(e.message);
        } finally {
            setLoading(false);
        }
    };

    const handleBatchDemo = async () => {
        setBatchResult(null);
        setError(null);
        try {
            const res = await fetch(`${API_BASE}/api/v1/predict/batch`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    texts: [
                        'Hello how are you?',
                        'WIN FREE CASH NOW!!!',
                        'Meeting tomorrow at 10',
                    ]
                }),
            });
            if (!res.ok) {
                const detail = await res.json();
                throw new Error(detail.detail || 'Batch prediction failed');
            }
            setBatchResult(await res.json());
        } catch (e) {
            setError(e.message);
        }
    };

    return (
        <div className="App">
            <header className="App-header">
                <h1>Spam Detection Demo</h1>
            </header>
            <main className="App-main">
                <section className="panel">
                    <h2>Status</h2>
                    {health ? (
                        <ul className="status-list">
                            <li>Service: <strong>{health.status}</strong></li>
                            <li>Model Loaded: <strong>{health.model_loaded ? 'Yes' : 'No'}</strong></li>
                        </ul>
                    ) : <p>Loading health...</p>}
                    {modelInfo && (
                        <div className="model-info">
                            <p><strong>Model:</strong> {modelInfo.model_name} ({modelInfo.model_type})</p>
                            <p><strong>Total Features:</strong> {modelInfo.features.total_features}</p>
                        </div>
                    )}
                    {error && <div className="error">{error}</div>}
                </section>
                <section className="panel">
                    <h2>Single Prediction</h2>
                    <form onSubmit={handlePredict} className="predict-form">
                        <textarea
                            value={text}
                            onChange={(e) => setText(e.target.value)}
                            placeholder="Enter text to classify..."
                            rows={4}
                        />
                        <div className="actions">
                            <button type="submit" disabled={loading || !text.trim()}> {loading ? 'Predicting...' : 'Predict'} </button>
                        </div>
                    </form>
                    {prediction && (
                        <div className={`result ${prediction.is_spam ? 'spam' : 'ham'}`}>
                            <p><strong>Prediction:</strong> {prediction.prediction.toUpperCase()}</p>
                            <p><strong>Confidence:</strong> {(prediction.confidence * 100).toFixed(2)}%</p>
                            <p><strong>Spam Prob:</strong> {(prediction.spam_probability * 100).toFixed(2)}% | <strong>Ham Prob:</strong> {(prediction.ham_probability * 100).toFixed(2)}%</p>
                        </div>
                    )}
                </section>
                <section className="panel">
                    <h2>Batch Prediction Demo</h2>
                    <p>This demonstrates the batch endpoint without building full UI.</p>
                    <button onClick={handleBatchDemo}>Run Batch Demo</button>
                    {batchResult && (
                        <div className="batch-results">
                            <p><strong>Total:</strong> {batchResult.total}</p>
                            <ul>
                                {batchResult.predictions.map((p, i) => (
                                    <li key={i} className={p.is_spam ? 'spam' : 'ham'}>
                                        {p.text.slice(0, 40)} - {p.prediction.toUpperCase()} ({(p.confidence * 100).toFixed(1)}%)
                                    </li>
                                ))}
                            </ul>
                        </div>
                    )}
                </section>
            </main>
        </div>
    );
}

export default SpamDemo;
