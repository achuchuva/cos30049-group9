import React, { useState, useEffect } from 'react';
import Tabs from '../components/Tabs';

const API_BASE = 'http://localhost:8000';

export default function About() {
	const [tab, setTab] = useState('MODEL');
	const [modelInfo, setModelInfo] = useState(null);
	const [loading, setLoading] = useState(true);
	const [error, setError] = useState(null);

	useEffect(() => {
		fetch(`${API_BASE}/api/v1/model/info`)
			.then(res => res.json())
			.then(data => {
				setModelInfo(data);
				setLoading(false);
			})
			.catch(err => {
				console.error(err);
				setError('Failed to load model info');
				setLoading(false);
			});
	}, []);

	return (
		<div>
			<section className="hero">
				<div className="hero-content">
					<h1 className="page-title">About</h1>
					<div className="team-info">
						<p className="team-lead">Group 9 • Session 22</p>
						<p className="team-tutor">Tutor: Yinwei Bao</p>
						<div className="team-members">
							<span className="member">Anton Chucuva</span>
							<span className="member-sep">•</span>
							<span className="member">Joshua Causon</span>
							<span className="member-sep">•</span>
							<span className="member">Kim Thongdee</span>
						</div>
					</div>
				</div>
				<div className="hero-tabs">
					<Tabs tabs={['MODEL', 'API', 'FRONTEND']} current={tab} onChange={setTab} />
				</div>
			</section>

			<div className="container">
				{tab === 'MODEL' && (
					<>
						<h2>Machine Learning Model</h2>
						{loading && <p>Loading model information...</p>}
						{error && <div className="error">{error}</div>}
						{modelInfo && (
							<>
								<div style={{ marginBottom: '1.5rem' }}>
									<h3>Model Details</h3>
									<ul style={{ lineHeight: '1.8' }}>
										<li><strong>Model Name:</strong> {modelInfo.model_name}</li>
										<li><strong>Algorithm:</strong> {modelInfo.model_type}</li>
										<li><strong>Total Features:</strong> {modelInfo.features.total_features}</li>
										<li><strong>Text Features (TF-IDF):</strong> {modelInfo.features.text_features}</li>
										<li><strong>Numerical Features:</strong> {modelInfo.features.numerical_features}</li>
										<li><strong>Loaded At:</strong> {new Date(modelInfo.loaded_at).toLocaleString()}</li>
									</ul>
								</div>

								<div style={{ marginBottom: '1.5rem' }}>
									<h3>Features Used</h3>
									<p style={{ fontSize: '0.9rem', marginBottom: '0.5rem' }}>
										The model uses a combination of text and numerical features to classify messages:
									</p>
									<ul style={{ lineHeight: '1.8', fontSize: '0.9rem' }}>
										<li><strong>TF-IDF Vectorization:</strong> Converts text into {modelInfo.features.text_features} numerical features representing word importance</li>
										<li><strong>Character Count:</strong> Total number of characters in the message</li>
										<li><strong>Word Count:</strong> Total number of words in the message</li>
										<li><strong>Suspicious Word Count:</strong> Number of words commonly found in spam (e.g., "free", "winner", "click")</li>
										<li><strong>URL Count:</strong> Number of URLs detected in the message</li>
										<li><strong>URL Digit Count:</strong> Number of digits found in URLs (often indicative of spam)</li>
									</ul>
								</div>
							</>
						)}
					</>
				)}

				{tab === 'FRONTEND' && (
					<>
						<h2>Frontend Technologies</h2>
						<div style={{ marginBottom: '1.5rem' }}>
							<h3>Core Framework</h3>
							<ul style={{ lineHeight: '1.8' }}>
								<li><strong>React 19.2.0:</strong> Modern UI library with hooks and functional components</li>
								<li><strong>React Router DOM 6.30.1:</strong> Client-side routing for navigation between pages</li>
								<li><strong>React Scripts 5.0.1:</strong> Build tooling and development server</li>
							</ul>
						</div>

						<div style={{ marginBottom: '1.5rem' }}>
							<h3>Data Visualization Libraries</h3>
							<ul style={{ lineHeight: '1.8' }}>
								<li><strong>Plotly.js 3.2.0:</strong> Interactive confidence timeline charts</li>
								<li><strong>react-plotly.js 2.6.0:</strong> React wrapper for Plotly</li>
								<li><strong>Chart.js 4.5.1:</strong> Spam/Ham distribution bar charts</li>
								<li><strong>react-chartjs-2 5.3.1:</strong> React wrapper for Chart.js</li>
								<li><strong>D3.js:</strong> Feature scatter plots with custom SVG rendering</li>
							</ul>
						</div>

						<div>
							<h3>Additional Libraries</h3>
							<ul style={{ lineHeight: '1.8' }}>
								<li><strong>Axios 1.13.2:</strong> HTTP client for API requests</li>
								<li><strong>JSZip 3.10.1:</strong> File handling for uploads</li>
							</ul>
						</div>
					</>
				)}

				{tab === 'API' && (
					<>
						<h2>API Endpoints</h2>
						<div style={{ marginBottom: '1.5rem' }}>
							<h3>Base URL</h3>
							<p style={{ fontFamily: 'monospace', fontSize: '0.9rem', background: '#f5f5f5', padding: '8px 12px', borderRadius: '4px' }}>
								http://localhost:8000
							</p>
						</div>

						<div style={{ marginBottom: '1.5rem' }}>
							<h3>Health & Info</h3>
							
							<div style={{ marginBottom: '1.2rem', paddingLeft: '1rem', borderLeft: '3px solid #4a90e2' }}>
								<h4 style={{ marginBottom: '0.5rem', fontSize: '1rem' }}>GET /</h4>
								<p style={{ fontSize: '0.9rem', color: '#666', marginBottom: '0' }}>Root endpoint with API information and available endpoints</p>
							</div>

							<div style={{ marginBottom: '1.2rem', paddingLeft: '1rem', borderLeft: '3px solid #4a90e2' }}>
								<h4 style={{ marginBottom: '0.5rem', fontSize: '1rem' }}>GET /health</h4>
								<p style={{ fontSize: '0.9rem', color: '#666', marginBottom: '0' }}>Health check endpoint to verify API and model availability</p>
							</div>

							<div style={{ marginBottom: '1.2rem', paddingLeft: '1rem', borderLeft: '3px solid #4a90e2' }}>
								<h4 style={{ marginBottom: '0.5rem', fontSize: '1rem' }}>GET /api/v1/model/info</h4>
								<p style={{ fontSize: '0.9rem', color: '#666', marginBottom: '0' }}>Get model information and features (used by MODEL tab)</p>
							</div>
						</div>

						<div style={{ marginBottom: '1.5rem' }}>
							<h3>Prediction</h3>
							
							<div style={{ marginBottom: '1.2rem', paddingLeft: '1rem', borderLeft: '3px solid #1a8f3a' }}>
								<h4 style={{ marginBottom: '0.5rem', fontSize: '1rem' }}>POST /api/v1/predict</h4>
								<p style={{ fontSize: '0.9rem', color: '#666', marginBottom: '0.5rem' }}>Classify a text message as spam or ham</p>
								<p style={{ fontSize: '0.85rem', marginBottom: '0.3rem' }}><strong>Request Body:</strong></p>
								<pre style={{ background: '#f5f5f5', padding: '10px', borderRadius: '4px', fontSize: '0.8rem', overflow: 'auto', margin: '0 0 0.5rem 0' }}>
{`{ "text": "Your message text here" }`}
								</pre>
								<p style={{ fontSize: '0.85rem', marginBottom: '0.3rem' }}><strong>Response:</strong></p>
								<pre style={{ background: '#f5f5f5', padding: '10px', borderRadius: '4px', fontSize: '0.8rem', overflow: 'auto', margin: '0' }}>
{`{
  "prediction": "spam" | "ham",
  "is_spam": true | false,
  "confidence": 0.95,
  "spam_probability": 0.95,
  "ham_probability": 0.05,
  "features": { ... },
  "timestamp": "2025-11-08T...",
  "model_name": "Logistic Regression",
  "prediction_id": 1
}`}
								</pre>
							</div>

							<div style={{ paddingLeft: '1rem', borderLeft: '3px solid #1a8f3a' }}>
								<h4 style={{ marginBottom: '0.5rem', fontSize: '1rem' }}>POST /api/v1/predict/file</h4>
								<p style={{ fontSize: '0.9rem', color: '#666', marginBottom: '0.5rem' }}>Classify a file (TXT, PDF, DOCX) as spam or ham</p>
								<p style={{ fontSize: '0.85rem', marginBottom: '0.3rem' }}><strong>Request:</strong> multipart/form-data with file field</p>
								<p style={{ fontSize: '0.85rem', marginBottom: '0' }}><strong>Response:</strong> FileUploadResponse with prediction_result matching /predict response</p>
							</div>
						</div>

						<div style={{ marginBottom: '1.5rem' }}>
							<h3>History & Statistics</h3>
							
							<div style={{ marginBottom: '1.2rem', paddingLeft: '1rem', borderLeft: '3px solid #e2a44a' }}>
								<h4 style={{ marginBottom: '0.5rem', fontSize: '1rem' }}>GET /api/v1/history</h4>
								<p style={{ fontSize: '0.9rem', color: '#666', marginBottom: '0.5rem' }}>Get prediction history with pagination</p>
								<p style={{ fontSize: '0.85rem', marginBottom: '0' }}><strong>Query params:</strong> limit (default 50, max 200), offset (default 0)</p>
							</div>

							<div style={{ marginBottom: '1.2rem', paddingLeft: '1rem', borderLeft: '3px solid #e2a44a' }}>
								<h4 style={{ marginBottom: '0.5rem', fontSize: '1rem' }}>DELETE /api/v1/history/:prediction_id</h4>
								<p style={{ fontSize: '0.9rem', color: '#666', marginBottom: '0' }}>Delete a specific prediction from history</p>
							</div>

							<div style={{ paddingLeft: '1rem', borderLeft: '3px solid #e2a44a' }}>
								<h4 style={{ marginBottom: '0.5rem', fontSize: '1rem' }}>GET /api/v1/stats</h4>
								<p style={{ fontSize: '0.9rem', color: '#666', marginBottom: '0' }}>Get aggregated statistics for visualizations (used by Results page)</p>
							</div>
						</div>

						<div style={{ marginBottom: '1.5rem' }}>
							<h3>Data Export</h3>
							
							<div style={{ paddingLeft: '1rem', borderLeft: '3px solid #9b59b6' }}>
								<h4 style={{ marginBottom: '0.5rem', fontSize: '1rem' }}>GET /api/v1/export/:format</h4>
								<p style={{ fontSize: '0.9rem', color: '#666', marginBottom: '0.5rem' }}>Export all predictions in CSV or JSON format</p>
								<p style={{ fontSize: '0.85rem', marginBottom: '0' }}><strong>Formats:</strong> csv, json</p>
							</div>
						</div>

						<div>
							<h3>Admin Operations</h3>
							
							<div style={{ marginBottom: '1.2rem', paddingLeft: '1rem', borderLeft: '3px solid #e03232' }}>
								<h4 style={{ marginBottom: '0.5rem', fontSize: '1rem' }}>PUT /api/v1/model/reload</h4>
								<p style={{ fontSize: '0.9rem', color: '#666', marginBottom: '0' }}>Reload the machine learning model without restarting the server</p>
							</div>

							<div style={{ paddingLeft: '1rem', borderLeft: '3px solid #e03232' }}>
								<h4 style={{ marginBottom: '0.5rem', fontSize: '1rem' }}>DELETE /api/v1/cache/clear</h4>
								<p style={{ fontSize: '0.9rem', color: '#666', marginBottom: '0' }}>Clear internal caches (placeholder for future implementation)</p>
							</div>
						</div>
					</>
				)}
			</div>
		</div>
	);
}
