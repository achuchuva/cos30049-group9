import React, { useState } from 'react';
import axios from 'axios';
import Tabs from '../components/Tabs';
import Chartjs from '../components/ChartjsImplementation';
import Plotly from '../components/PlotlyImplementation';
import D3 from '../components/D3Implementation';
import ChartjsDoughnut from '../components/ChartjsDoughnut';
import ChartjsBar from '../components/ChartjsBar';
import PlotlyBar from '../components/PlotlyBar';
import PlotlyLine from '../components/PlotlyLine';

const API_BASE_URL = 'http://localhost:8000';

export default function Results() {
	const [tab, setTab] = useState('PLOTLY');
	const [exporting, setExporting] = useState(false);

	const handleExport = async (format) => {
		try {
			setExporting(true);
			const response = await axios.get(`${API_BASE_URL}/api/v1/export/${format}`, {
				responseType: 'blob'
			});

			// Create download link
			const url = window.URL.createObjectURL(new Blob([response.data]));
			const link = document.createElement('a');
			link.href = url;
			link.setAttribute('download', `predictions.${format}`);
			document.body.appendChild(link);
			link.click();
			link.parentNode.removeChild(link);
			window.URL.revokeObjectURL(url);
		} catch (error) {
			console.error(`Error exporting ${format.toUpperCase()}:`, error);
			alert(`Failed to export ${format.toUpperCase()}. Please ensure there is prediction history available.`);
		} finally {
			setExporting(false);
		}
	};

	return (
		<div>
			<section className="hero">
				<div className="hero-content">
					<h1 className="page-title">Results</h1>
					<div style={{ 
						display: 'flex', 
						gap: '12px', 
						marginTop: '16px',
						justifyContent: 'center'
					}}>
						<button 
							className="button-primary" 
							onClick={() => handleExport('csv')}
							disabled={exporting}
							style={{ fontSize: '14px', padding: '8px 20px' }}
						>
							{exporting ? 'Exporting...' : 'Export CSV'}
						</button>
						<button 
							className="button-primary" 
							onClick={() => handleExport('json')}
							disabled={exporting}
							style={{ fontSize: '14px', padding: '8px 20px' }}
						>
							{exporting ? 'Exporting...' : 'Export JSON'}
						</button>
					</div>
				</div>
				<div className="hero-tabs">
					<Tabs tabs={['PLOTLY', 'CHART', 'D3']} current={tab} onChange={setTab} />
				</div>
			</section>

			<div className="container">
				{tab === 'PLOTLY' && (
					<>
						<Plotly />
						<PlotlyBar />
					</>
				)}
				{tab === 'CHART' && (
					<Chartjs />
				)}
				{tab === 'D3' && (
					<D3 />
				)}
			</div>
		</div>
	);
}
