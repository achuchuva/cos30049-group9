import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, ArcElement, Tooltip, Legend } from 'chart.js';
import { Bar, Doughnut } from 'react-chartjs-2';
import axios from 'axios';
import { useEffect, useState, useRef } from 'react';

const API_BASE = 'http://localhost:8000';

ChartJS.register(CategoryScale, LinearScale, BarElement, ArcElement, Tooltip, Legend);

function Chartjs() {
	const [stats, setStats] = useState(null);
	const [error, setError] = useState(null);
	const barChartRef = useRef(null);
	const doughnutChartRef = useRef(null);

	useEffect(() => {
		axios.get(`${API_BASE}/api/v1/stats`)
			.then((response) => {
				if (response.status === 200) {
					console.log(response.data);
					setStats(response.data);
				}
				else {
					setError("ERROR: " + response.status + " , " + response.statusText);
				}
			})
			.catch((err) => {
				console.error(err);
				setError("Failed to load stats. Please ensure the backend server is running.");
			});
	}, []);

	const exportChart = (chartRef, filename) => {
		if (chartRef.current) {
			const url = chartRef.current.toBase64Image();
			const link = document.createElement('a');
			link.download = filename;
			link.href = url;
			link.click();
		}
	};

	if (error) {
		return <div className="error">{error}</div>;
	}

	if (!stats) {
		return <div>Loading...</div>;
	}

	const barData = {
		labels: ['Spam', 'Ham'],
		datasets: [
			{
				label: 'Prediction Count',
				data: [stats.spam_count, stats.ham_count],
				backgroundColor: [
					'rgba(231, 45, 12, 0.82)',
					'rgba(125, 255, 86, 0.82)',
				],
				borderColor: [
					'rgba(235, 54, 54, 1)',
					'rgba(75, 192, 75, 1)',
				],
				borderWidth: 2,
			},
		],
	};

	const barOptions = {
		responsive: true,
		maintainAspectRatio: true,
		plugins: {
			legend: {
				display: false,
			},
			title: {
				display: true,
				text: 'Spam vs Ham Distribution',
				font: {
					size: 18,
					weight: 'bold',
				},
			},
		},
		scales: {
			y: {
				beginAtZero: true,
				title: {
					display: true,
					text: 'Count',
					font: {
						size: 14,
						weight: 'bold',
					},
				},
			},
			x: {
				title: {
					display: true,
					text: 'Classification',
					font: {
						size: 14,
						weight: 'bold',
					},
				},
			},
		},
	};

	const doughnutData = {
		labels: ['Spam', 'Ham'],
		datasets: [
			{
				label: 'Distribution',
				data: [stats.spam_count, stats.ham_count],
				backgroundColor: [
					'rgba(231, 45, 12, 0.7)',
					'rgba(125, 255, 86, 0.7)',
				],
				borderColor: [
					'rgba(235, 54, 54, 1)',
					'rgba(75, 192, 75, 1)',
				],
				borderWidth: 3,
			},
		],
	};

	const doughnutOptions = {
		responsive: true,
		maintainAspectRatio: true,
		plugins: {
			legend: {
				display: true,
				position: 'bottom',
				labels: {
					font: {
						size: 14,
					},
					padding: 20,
				},
			},
			title: {
				display: true,
				text: 'Spam Rate Overview',
				font: {
					size: 18,
					weight: 'bold',
				},
			},
			tooltip: {
				callbacks: {
					label: function(context) {
						const label = context.label || '';
						const value = context.parsed || 0;
						const total = stats.total_predictions;
						const percentage = ((value / total) * 100).toFixed(1);
						return `${label}: ${value} (${percentage}%)`;
					}
				}
			},
		},
	};

	return (
		<section>
			<div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
				<h2 style={{ margin: 0 }}>Prediction Analytics</h2>
				<div style={{ display: 'flex', gap: '10px' }}>
					<button 
						className="btn" 
						onClick={() => exportChart(barChartRef, 'spam-distribution-bar.png')}
						style={{ fontSize: '0.85rem' }}
					>
						Export Bar Chart
					</button>
					<button 
						className="btn" 
						onClick={() => exportChart(doughnutChartRef, 'spam-distribution-doughnut.png')}
						style={{ fontSize: '0.85rem' }}
					>
						Export Doughnut Chart
					</button>
				</div>
			</div>
			
			<div style={{ 
				display: 'grid', 
				gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', 
				gap: '20px',
				marginBottom: '2rem'
			}}>
				<div style={{ 
					padding: '16px', 
					background: '#f9f9f9', 
					borderRadius: '8px', 
					border: '1px solid #e5e5e5' 
				}}>
					<h4 style={{ margin: '0 0 8px 0', fontSize: '0.9rem', color: '#666' }}>Total Predictions</h4>
					<p style={{ margin: 0, fontSize: '2rem', fontWeight: 'bold', color: '#333' }}>
						{stats.total_predictions}
					</p>
				</div>
				<div style={{ 
					padding: '16px', 
					background: '#fff5f5', 
					borderRadius: '8px', 
					border: '1px solid #ffcccb' 
				}}>
					<h4 style={{ margin: '0 0 8px 0', fontSize: '0.9rem', color: '#666' }}>Spam Count</h4>
					<p style={{ margin: 0, fontSize: '2rem', fontWeight: 'bold', color: '#e03232' }}>
						{stats.spam_count}
					</p>
				</div>
				<div style={{ 
					padding: '16px', 
					background: '#f0fff4', 
					borderRadius: '8px', 
					border: '1px solid #b2f5c2' 
				}}>
					<h4 style={{ margin: '0 0 8px 0', fontSize: '0.9rem', color: '#666' }}>Ham Count</h4>
					<p style={{ margin: 0, fontSize: '2rem', fontWeight: 'bold', color: '#1a8f3a' }}>
						{stats.ham_count}
					</p>
				</div>
				<div style={{ 
					padding: '16px', 
					background: '#f4f4f4', 
					borderRadius: '8px', 
					border: '1px solid #d5d5d5' 
				}}>
					<h4 style={{ margin: '0 0 8px 0', fontSize: '0.9rem', color: '#666' }}>Spam Rate</h4>
					<p style={{ margin: 0, fontSize: '2rem', fontWeight: 'bold', color: '#4a90e2' }}>
						{(stats.spam_rate * 100).toFixed(1)}%
					</p>
				</div>
			</div>

			<div style={{ 
				display: 'grid', 
				gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', 
				gap: '30px',
				marginTop: '2rem'
			}}>
				<div>
					<Bar ref={barChartRef} data={barData} options={barOptions} />
				</div>
				<div style={{ maxWidth: '400px', margin: '0 auto' }}>
					<Doughnut ref={doughnutChartRef} data={doughnutData} options={doughnutOptions} />
				</div>
			</div>
		</section>
	);
}

export default Chartjs;