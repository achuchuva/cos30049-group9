import './App.css';
import { useEffect, useState } from 'react';

const API_BASE = 'http://localhost:8000';

function SpamDemo() {
    const [health, setHealth] = useState(null);
    const [modelInfo, setModelInfo] = useState(null);
    const [text, setText] = useState('');
    const [prediction, setPrediction] = useState(null);
    const [file, setFile] = useState(null);
    const [fileResult, setFileResult] = useState(null);
    const [history, setHistory] = useState(null);
    const [stats, setStats] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [historyOffset, setHistoryOffset] = useState(0);
    const historyLimit = 5;

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
            fetchHistory();
        } catch (e) {
            setError(e.message);
        } finally {
            setLoading(false);
        }
    };

    const handleFileUpload = async (e) => {
        e.preventDefault();
        if (!file) return;
        setLoading(true);
        setError(null);
        setFileResult(null);
        try {
            const formData = new FormData();
            formData.append('file', file);
            const res = await fetch(`${API_BASE}/api/v1/predict/file`, {
                method: 'POST',
                body: formData,
            });
            if (!res.ok) {
                const detail = await res.json();
                throw new Error(detail.detail || 'File upload failed');
            }
            setFileResult(await res.json());
            fetchHistory();
        } catch (e) {
            setError(e.message);
        } finally {
            setLoading(false);
        }
    };

    const fetchHistory = async (offset = 0) => {
        try {
            const res = await fetch(`${API_BASE}/api/v1/history?limit=${historyLimit}&offset=${offset}`);
            if (res.ok) {
                setHistory(await res.json());
                setHistoryOffset(offset);
            }
        } catch (e) {
            console.error('Failed to fetch history:', e);
        }
    };

    const handleDeletePrediction = async (id) => {
        try {
            const res = await fetch(`${API_BASE}/api/v1/history/${id}`, {
                method: 'DELETE',
            });
            if (res.ok) {
                fetchHistory(historyOffset);
            }
        } catch (e) {
            setError('Failed to delete prediction');
        }
    };

    const fetchStats = async () => {
        try {
            const res = await fetch(`${API_BASE}/api/v1/stats`);
            if (res.ok) {
                setStats(await res.json());
            }
        } catch (e) {
            setError('Failed to fetch stats');
        }
    };

    const handleExport = async (format) => {
        try {
            const res = await fetch(`${API_BASE}/api/v1/export/${format}`);
            if (res.ok) {
                const blob = await res.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `predictions.${format}`;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
            }
        } catch (e) {
            setError(`Failed to export as ${format}`);
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
                    <h2>Single Text Prediction</h2>
                    <form onSubmit={handlePredict} className="predict-form">
                        <textarea
                            value={text}
                            onChange={(e) => setText(e.target.value)}
                            placeholder="Enter text to classify..."
                            rows={4}
                        />
                        <div className="actions">
                            <button type="submit" disabled={loading || !text.trim()}>
                                {loading ? 'Predicting...' : 'Predict'}
                            </button>
                        </div>
                    </form>
                    {prediction && (
                        <div className={`result ${prediction.is_spam ? 'spam' : 'ham'}`}>
                            <p><strong>Prediction:</strong> {prediction.prediction.toUpperCase()}</p>
                            <p><strong>Confidence:</strong> {(prediction.confidence * 100).toFixed(2)}%</p>
                            <p><strong>Spam Prob:</strong> {(prediction.spam_probability * 100).toFixed(2)}% | <strong>Ham Prob:</strong> {(prediction.ham_probability * 100).toFixed(2)}%</p>
                            {prediction.features && (
                                <div>
                                    <p><strong>Features:</strong></p>
                                    <ul style={{fontSize: '0.8rem', marginTop: '0.3rem'}}>
                                        <li>Characters: {prediction.features.char_count}</li>
                                        <li>Words: {prediction.features.word_count}</li>
                                        <li>Suspicious words: {prediction.features.suspicious_word_count}</li>
                                        <li>URLs: {prediction.features.url_count}</li>
                                        <li>URL digits: {prediction.features.url_digit_count}</li>
                                    </ul>
                                </div>
                            )}
                            {prediction.prediction_id && (
                                <p style={{fontSize: '0.75rem', marginTop: '0.5rem'}}>
                                    <em>Saved as ID: {prediction.prediction_id}</em>
                                </p>
                            )}
                        </div>
                    )}
                </section>

                <section className="panel">
                    <h2>File Upload Prediction</h2>
                    <p style={{fontSize: '0.85rem', color: '#666'}}>Upload TXT, PDF, or DOCX files (max 5MB)</p>
                    <form onSubmit={handleFileUpload} className="predict-form">
                        <input
                            type="file"
                            accept=".txt,.pdf,.docx"
                            onChange={(e) => setFile(e.target.files[0])}
                            style={{marginBottom: '0.5rem'}}
                        />
                        <div className="actions">
                            <button type="submit" disabled={loading || !file}>
                                {loading ? 'Uploading...' : 'Upload & Predict'}
                            </button>
                        </div>
                    </form>
                    {fileResult && (
                        <div>
                            <p style={{fontSize: '0.85rem'}}>
                                <strong>File:</strong> {fileResult.filename} ({(fileResult.file_size_bytes / 1024).toFixed(1)} KB)
                            </p>
                            <p style={{fontSize: '0.85rem'}}>
                                <strong>Extracted:</strong> {fileResult.extracted_text_length} characters
                            </p>
                            <div className={`result ${fileResult.prediction_result.is_spam ? 'spam' : 'ham'}`}>
                                <p><strong>Prediction:</strong> {fileResult.prediction_result.prediction.toUpperCase()}</p>
                                <p><strong>Confidence:</strong> {(fileResult.prediction_result.confidence * 100).toFixed(2)}%</p>
                                <p><strong>Text preview:</strong> {fileResult.prediction_result.text.substring(0, 100)}...</p>
                            </div>
                        </div>
                    )}
                </section>

                <section className="panel">
                    <h2>Prediction History</h2>
                    <div className="actions">
                        <button onClick={() => fetchHistory(0)}>Load History</button>
                        {history && history.total > 0 && (
                            <>
                                <button onClick={() => fetchHistory(Math.max(0, historyOffset - historyLimit))} disabled={historyOffset === 0}>
                                    Previous
                                </button>
                                <button onClick={() => fetchHistory(historyOffset + historyLimit)} disabled={historyOffset + historyLimit >= history.total}>
                                    Next
                                </button>
                            </>
                        )}
                    </div>
                    {history && (
                        <div>
                            <p style={{fontSize: '0.85rem', margin: '0.5rem 0'}}>
                                Showing {historyOffset + 1}-{Math.min(historyOffset + historyLimit, history.total)} of {history.total}
                            </p>
                            {history.predictions.length > 0 ? (
                                <ul style={{listStyle: 'none', padding: 0}}>
                                    {history.predictions.map((p) => (
                                        <li key={p.id} className={`result ${p.is_spam ? 'spam' : 'ham'}`} style={{marginBottom: '0.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
                                            <div style={{flex: 1}}>
                                                <p style={{margin: '0.2rem 0'}}><strong>#{p.id}</strong> - {p.prediction.toUpperCase()} ({(p.confidence * 100).toFixed(1)}%)</p>
                                                <p style={{margin: '0.2rem 0', fontSize: '0.75rem'}}>{p.text_preview.substring(0, 60)}...</p>
                                                <p style={{margin: '0.2rem 0', fontSize: '0.7rem', color: '#666'}}>
                                                    {p.source_type} {p.filename && `(${p.filename})`} - {new Date(p.timestamp).toLocaleString()}
                                                </p>
                                            </div>
                                            <button onClick={() => handleDeletePrediction(p.id)} style={{marginLeft: '0.5rem', background: '#dc3545'}}>
                                                Delete
                                            </button>
                                        </li>
                                    ))}
                                </ul>
                            ) : (
                                <p style={{fontSize: '0.85rem', color: '#666'}}>No predictions in history</p>
                            )}
                        </div>
                    )}
                </section>

                <section className="panel">
                    <h2>Statistics</h2>
                    <button onClick={fetchStats}>Load Statistics</button>
                    {stats && (
                        <div style={{marginTop: '0.5rem'}}>
                            <div style={{display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '0.5rem', marginBottom: '1rem'}}>
                                <div style={{background: '#f0f0f0', padding: '0.5rem', borderRadius: '4px'}}>
                                    <p style={{margin: 0, fontSize: '0.75rem', color: '#666'}}>Total Predictions</p>
                                    <p style={{margin: 0, fontSize: '1.5rem', fontWeight: 'bold'}}>{stats.total_predictions}</p>
                                </div>
                                <div style={{background: '#ffe5e5', padding: '0.5rem', borderRadius: '4px'}}>
                                    <p style={{margin: 0, fontSize: '0.75rem', color: '#666'}}>Spam Count</p>
                                    <p style={{margin: 0, fontSize: '1.5rem', fontWeight: 'bold'}}>{stats.spam_count}</p>
                                </div>
                                <div style={{background: '#e5f7e5', padding: '0.5rem', borderRadius: '4px'}}>
                                    <p style={{margin: 0, fontSize: '0.75rem', color: '#666'}}>Ham Count</p>
                                    <p style={{margin: 0, fontSize: '1.5rem', fontWeight: 'bold'}}>{stats.ham_count}</p>
                                </div>
                                <div style={{background: '#e5f0ff', padding: '0.5rem', borderRadius: '4px'}}>
                                    <p style={{margin: 0, fontSize: '0.75rem', color: '#666'}}>Spam Rate</p>
                                    <p style={{margin: 0, fontSize: '1.5rem', fontWeight: 'bold'}}>{(stats.spam_rate * 100).toFixed(1)}%</p>
                                </div>
                            </div>
                            <div>
                                <p style={{fontSize: '0.85rem', fontWeight: 'bold', marginBottom: '0.3rem'}}>Feature Averages:</p>
                                <ul style={{fontSize: '0.8rem', marginTop: '0.3rem'}}>
                                    <li>Avg characters: {stats.feature_averages.avg_char_count?.toFixed(1) || 'N/A'}</li>
                                    <li>Avg words: {stats.feature_averages.avg_word_count?.toFixed(1) || 'N/A'}</li>
                                    <li>Avg suspicious words: {stats.feature_averages.avg_suspicious_words?.toFixed(1) || 'N/A'}</li>
                                    <li>Avg URLs: {stats.feature_averages.avg_url_count?.toFixed(1) || 'N/A'}</li>
                                </ul>
                            </div>
                            <div>
                                <p style={{fontSize: '0.85rem', fontWeight: 'bold', marginBottom: '0.3rem'}}>Avg Confidence: {(stats.avg_confidence * 100).toFixed(2)}%</p>
                            </div>
                        </div>
                    )}
                </section>

                <section className="panel">
                    <h2>Export Data</h2>
                    <p style={{fontSize: '0.85rem', color: '#666'}}>Download all prediction history</p>
                    <div className="actions">
                        <button onClick={() => handleExport('json')}>Export as JSON</button>
                        <button onClick={() => handleExport('csv')}>Export as CSV</button>
                    </div>
                </section>
            </main>
        </div>
    );
}

export default SpamDemo;
