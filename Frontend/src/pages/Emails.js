import React, { useState, useEffect } from 'react';

const API_BASE = 'http://localhost:8000';

export default function Emails() {
    const [predictions, setPredictions] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        const fetchPredictions = async () => {
            try {
                setLoading(true);
                const res = await fetch(`${API_BASE}/api/v1/email/predictions?limit=100`);
                if (res.ok) {
                    const data = await res.json();
                    setPredictions(data.predictions || []);
                } else {
                    setError('Failed to fetch email predictions.');
                }
            } catch (err) {
                setError('An error occurred while fetching data.');
                console.error(err);
            } finally {
                setLoading(false);
            }
        };

        fetchPredictions();
    }, []);

    return (
        <div className="emails-container">
            <h1>Detected Emails</h1>
            <p>This page displays emails that have been processed and classified as spam or ham.</p>

            {loading && <p>Loading...</p>}
            {error && <p className="error">{error}</p>}

            {!loading && !error && (
                <div className="email-list">
                    {predictions.length === 0 ? (
                        <p>No emails processed yet.</p>
                    ) : (
                        predictions.map(p => (
                            <div key={p.id} className={`email-item ${p.is_spam ? 'spam' : 'ham'}`}>
                                <div className="email-header">
                                    <span className="email-sender">{p.filename || 'Unknown Sender'}</span>
                                    <span className="email-timestamp">{new Date(p.timestamp).toLocaleString()}</span>
                                </div>
                                <div className="email-subject">{p.text_preview.split('\\n\\n')[0]}</div>
                                <div className="email-body">{p.text_preview.split('\\n\\n').slice(1).join('\\n\\n')}</div>
                                <div className="email-footer">
                                    <span className={`prediction-label ${p.is_spam ? 'spam' : 'ham'}`}>
                                        {p.prediction}
                                    </span>
                                    <span className="confidence">
                                        Confidence: {(p.confidence * 100).toFixed(1)}%
                                    </span>
                                </div>
                            </div>
                        ))
                    )}
                </div>
            )}
        </div>
    );
}
