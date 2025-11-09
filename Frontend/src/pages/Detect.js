import React, { useState } from 'react';
import Tabs from '../components/Tabs';

const API_BASE = 'http://localhost:8000';

export default function Detect() {
	const [tab, setTab] = useState('TEXT');
	const [text, setText] = useState('');
	const [file, setFile] = useState(null);
	const [fileName, setFileName] = useState('');
	const [prediction, setPrediction] = useState(null);
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState('');
	const [isDragging, setIsDragging] = useState(false);

	const handleKeyDown = (e) => {
		if (e.key === 'Enter') {
			if (e.shiftKey) {
				return;
			}

			// Prevent newline and submit on Enter
			e.preventDefault();
			if (text.trim()) {
				handlePredict(e);
			}
		}
	};

	const handlePredict = async (e) => {
		e.preventDefault();
		setLoading(true);
		setError('');
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
		} catch (err) {
			setError(err.message);
		} finally {
			setLoading(false);
		}
	};

	const handleFileUpload = async (e) => {
		e.preventDefault();
		if (!file) return;
		setLoading(true);
		setError('');
		setPrediction(null);

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
			const data = await res.json();
			setPrediction(data.prediction_result);
		} catch (err) {
			setError(err.message);
		} finally {
			setLoading(false);
		}
	};

	const handleDragEnter = (e) => {
		e.preventDefault();
		setIsDragging(true);
	};

	const handleDragLeave = (e) => {
		e.preventDefault();
		setIsDragging(false);
	};

	const handleDrop = (e) => {
		e.preventDefault();
		setIsDragging(false);
		const droppedFile = e.dataTransfer.files?.[0];
		if (droppedFile) {
			const ext = droppedFile.name.split('.').pop().toLowerCase();
			if (['txt', 'pdf', 'docx'].includes(ext)) {
				setFile(droppedFile);
				setFileName(droppedFile.name);
			} else {
				setError('Unsupported file type. Please upload a .txt, .pdf, or .docx file.');
			}
		}
	};

	return (
		<div>
			<section className="hero">
				<div className="hero-content">
					<h1 className="page-title">Spam Detection</h1>
				</div>
				<div className="hero-tabs">
					<Tabs tabs={['TEXT', 'FILE']} current={tab} onChange={setTab} />
				</div>
			</section>

			<div className="container">
				{error && (
					<div className="error" role="alert">
						<span className="error-emoji">❌</span>
						<strong>Error:</strong> {error}
					</div>
				)}

				{tab === 'TEXT' && (
					<form onSubmit={handlePredict} className="detect-form">
						<div className="field">
							<label>Enter Text</label>
							<textarea
								placeholder="Enter text to classify... (Press Enter to detect, Shift+Enter for new line)"
								rows={5}
								value={text}
								onChange={(e) => setText(e.target.value)}
								onKeyDown={handleKeyDown}
							/>
						</div>
						<button className="btn primary" disabled={loading || !text.trim()}>
							{loading ? 'Detecting...' : 'Detect'}
						</button>
					</form>
				)}

				{tab === 'FILE' && (
					<form onSubmit={handleFileUpload} className="detect-form">
						<div
							className={`dropzone ${isDragging ? 'dragging' : ''}`}
							onDragEnter={handleDragEnter}
							onDragOver={(e) => e.preventDefault()}
							onDragLeave={handleDragLeave}
							onDrop={handleDrop}
						>
							<input
								id="file"
								type="file"
								accept=".txt,.pdf,.docx"
								onChange={(e) => {
									const f = e.target.files?.[0];
									if (f) {
										setFile(f);
										setFileName(f.name);
										setError('');
									}
								}}
							/>
							<label htmlFor="file">
								<div className="dz-in">
									{fileName ? (
										<p className="file-selected">✅ {fileName}</p>
									) : (
										<>
											<p className="dropzone-text">
												Drag & drop file here or click to browse
											</p>
											<p className="supported-formats">
												Supported: .txt, .pdf, .docx
											</p>
										</>
									)}
								</div>
							</label>
						</div>
						<button className="btn primary" disabled={loading || !file}>
							{loading ? 'Detecting...' : 'Detect'}
						</button>
					</form>
				)}

				{prediction && (
					<div className={`result ${prediction.is_spam ? 'spam' : 'ham'}`}>
						<div className="result-header">
							{prediction.is_spam ? (
								<div>
									<span className="result-emoji">⚠️</span>
									<h3 className="result-heading">SPAM DETECTED</h3>
									<p className="result-subtext">This message appears to be spam or harmful content.</p>
								</div>
							) : (
								<div>
									<span className="result-emoji">✅</span>
									<h3 className="result-heading">NO SPAM DETECTED</h3>
									<p className="result-subtext">This message appears to be safe and legitimate.</p>
								</div>
							)}
						</div>
						<div className="result-details">
							<p><strong>Confidence:</strong> {(prediction.confidence * 100).toFixed(2)}%</p>
							<p>
								<strong>Spam Probability:</strong> {(prediction.spam_probability * 100).toFixed(2)}% |{' '}
								<strong>Ham Probability:</strong> {(prediction.ham_probability * 100).toFixed(2)}%
							</p>
							{prediction.features && (
								<div>
									<p><strong>Analysis Details:</strong></p>
									<ul style={{ fontSize: '0.85rem', marginTop: '0.3rem' }}>
										<li>Characters: {prediction.features.char_count}</li>
										<li>Words: {prediction.features.word_count}</li>
										<li>Suspicious words: {prediction.features.suspicious_word_count}</li>
										<li>URLs: {prediction.features.url_count}</li>
										<li>URL digits: {prediction.features.url_digit_count}</li>
									</ul>
								</div>
							)}
							{prediction.prediction_id && (
								<p style={{ fontSize: '0.75rem', marginTop: '0.5rem', opacity: 0.7 }}>
									<em>Saved as ID: {prediction.prediction_id}</em>
								</p>
							)}
						</div>
					</div>
				)}
			</div>
		</div>
	);
}
