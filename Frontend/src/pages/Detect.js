import React, { useMemo, useRef, useState } from 'react';
import Tabs from '../components/Tabs';

const API_BASE = 'http://localhost:8000/';

async function predictText(text, signal) {
  const res = await fetch(`${API_BASE}/api/v1/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
    signal,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function predictFile(file, signal) {
  const fd = new FormData();
  fd.append('file', file);
  const res = await fetch(`${API_BASE}/api/v1/predict/file`, {
    method: 'POST',
    body: fd,
    signal,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function mapToUIShape(raw) {
  const label = String(raw.prediction ?? raw.label ?? '');
  const conf = Number(raw.confidence ?? raw.score ?? 0);

  const spamProb =
    raw.spam_probability ??
    raw.prob_spam ??
    raw.label_probabilities?.spam ??
    raw.probs?.spam ??
    (label.toLowerCase() === 'spam' && conf ? conf : null);

  const hamProb =
    raw.ham_probability ??
    raw.prob_ham ??
    raw.label_probabilities?.ham ??
    raw.probs?.ham ??
    (typeof spamProb === 'number' ? 1 - spamProb : null);

  return {
    is_spam: label.toLowerCase() === 'spam',
    prediction: label,
    confidence: conf,
    spam_probability: typeof spamProb === 'number' ? spamProb : 0,
    ham_probability:
      typeof hamProb === 'number' ? hamProb : typeof spamProb === 'number' ? 1 - spamProb : 0,
    features: {
      char_count: raw.features?.char_count ?? raw.char_count ?? null,
      word_count: raw.features?.word_count ?? raw.word_count ?? null,
      suspicious_word_count: raw.features?.suspicious_word_count ?? raw.suspicious_word_count ?? null,
      url_count: raw.features?.url_count ?? raw.url_count ?? null,
      url_digit_count: raw.features?.url_digit_count ?? raw.url_digit_count ?? null,
    },
    prediction_id: raw.prediction_id ?? raw.id ?? null,
  };
}

export default function Detect() {
  const [tab, setTab] = useState('FILE');
  const [text, setText] = useState('');
  const [url, setUrl] = useState('');
  const [fileName, setFileName] = useState('');
  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const ctrlRef = useRef(null);

  const canSubmit = useMemo(() => {
    if (tab === 'FILE') return Boolean(file);
    if (tab === 'TEXT') return text.trim().length > 0;
    if (tab === 'URL') return url.trim().length > 0;
    return false;
  }, [tab, text, url, file]);

  const onFile = (e) => {
    const f = e.target.files?.[0];
    if (f) {
      setFile(f);
      setFileName(f.name);
    } else {
      setFile(null);
      setFileName('');
    }
  };

  const onSubmit = async (e) => {
    e.preventDefault();
    if (!canSubmit || loading) return;

    setLoading(true);
    setError('');
    setPrediction(null);

    if (ctrlRef.current) ctrlRef.current.abort();
    const controller = new AbortController();
    ctrlRef.current = controller;

    try {
      let data;
      if (tab === 'FILE' && file) data = await predictFile(file, controller.signal);
      else if (tab === 'TEXT') data = await predictText(text, controller.signal);
      else data = await predictText(url, controller.signal);

      setPrediction(mapToUIShape(data));
    } catch (err) {
      setError(err?.message || 'Request failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <section className="hero">
        <h1 className="page-title">Spam or Malware Detection</h1>
        <Tabs tabs={['FILE', 'TEXT', 'URL']} current={tab} onChange={setTab} />
      </section>

      <div className="container">
        <form onSubmit={onSubmit} className="detect-form">
          {tab === 'TEXT' && (
            <div className="field">
              <label>Spam Detection</label>
              <textarea
                placeholder="Enter potential spam"
                rows={5}
                value={text}
                onChange={(e) => setText(e.target.value)}
              />
            </div>
          )}

          {tab === 'FILE' && (
            <div className="dropzone">
              <input id="file" type="file" onChange={onFile} />
              <label htmlFor="file">
                <div className="dz-in">
                  <p>
                    Drag and drop your files
                    <br />
                    here to upload
                  </p>
                  <div className="or">OR</div>
                  <button type="button" className="btn">Browse Files</button>
                </div>
              </label>
              {fileName && (
                <div className="upload-msg">
                  File Uploaded: <strong>{fileName}</strong>
                </div>
              )}
            </div>
          )}

          {tab === 'URL' && (
            <div className="field">
              <label>URL</label>
              <input
                placeholder="https://example.com"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
              />
            </div>
          )}

          <button className="btn primary" disabled={!canSubmit || loading}>
            {loading ? 'Analyzing…' : 'Upload'}
          </button>
        </form>

        {prediction && (
          <div className={`result ${prediction.is_spam ? 'spam' : 'ham'}`}>
            <p><strong>Prediction:</strong> {prediction.prediction.toUpperCase()}</p>
            <p><strong>Confidence:</strong> {(prediction.confidence * 100).toFixed(2)}%</p>
            <p>
              <strong>Spam Prob:</strong> {(prediction.spam_probability * 100).toFixed(2)}% |{' '}
              <strong>Ham Prob:</strong> {(prediction.ham_probability * 100).toFixed(2)}%
            </p>
            {prediction.features && (
              <div>
                <p><strong>Features:</strong></p>
                <ul style={{ fontSize: '0.8rem', marginTop: '0.3rem' }}>
                  <li>Characters: {prediction.features.char_count}</li>
                  <li>Words: {prediction.features.word_count}</li>
                  <li>Suspicious words: {prediction.features.suspicious_word_count}</li>
                  <li>URLs: {prediction.features.url_count}</li>
                  <li>URL digits: {prediction.features.url_digit_count}</li>
                </ul>
              </div>
            )}
            {prediction.prediction_id && (
              <p style={{ fontSize: '0.75rem', marginTop: '0.5rem' }}>
                <em>Saved as ID: {prediction.prediction_id}</em>
              </p>
            )}
          </div>
        )}

        {error && <div className="error" role="alert">{error}</div>}
      </div>
    </div>
  );
}
