import React, { useEffect, useState } from 'react';
import Plot from 'react-plotly.js';
import axios from 'axios';

const API_BASE = 'http://localhost:8000';

export default function PlotlyImplementation() {
    const [stats, setStats] = useState(null);
    const [error, setError] = useState(null);

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

    if (error) {
        return <div className="error">{error}</div>;
    }

    if (!stats || !stats.confidence_distribution) {
        return <div>Loading...</div>;
    }

    // Reverse the array to show oldest first and give it a proper index
    const reversedConfidenceDistribution = [...stats.confidence_distribution].reverse();

    // Separate spam and ham predictions for different traces
    const spamData = reversedConfidenceDistribution
        .map((item, idx) => ({ ...item, idx }))
        .filter(item => item.prediction === 'spam');
    const hamData = reversedConfidenceDistribution
        .map((item, idx) => ({ ...item, idx }))
        .filter(item => item.prediction === 'ham');

    return (
        <div>
            <h2>Prediction Confidence Timeline</h2>
            <p style={{ fontSize: '0.85rem', color: '#666', marginBottom: '1rem' }}>
                Confidence scores for each prediction over time (hover for details, use toolbar to export)
            </p>
            <div style={{ display: 'flex', justifyContent: 'center' }}>
                <Plot
                    data={[
                        {
                            x: spamData.map(item => item.idx),
                            y: spamData.map(item => item.confidence),
                            type: 'scatter',
                            mode: 'lines+markers',
                            marker: { 
                                color: '#e72d0c', 
                                size: 8,
                                line: {
                                    color: '#b32209',
                                    width: 2
                                }
                            },
                            line: { color: '#e72d0c', width: 3 },
                            name: 'Spam',
                            hovertemplate: '<b>Spam</b><br>Prediction #%{x}<br>Confidence: %{y:.2%}<extra></extra>',
                        },
                        {
                            x: hamData.map(item => item.idx),
                            y: hamData.map(item => item.confidence),
                            type: 'scatter',
                            mode: 'lines+markers',
                            marker: { 
                                color: '#7dff56', 
                                size: 8,
                                line: {
                                    color: '#4fc22e',
                                    width: 2
                                }
                            },
                            line: { color: '#7dff56', width: 3 },
                            name: 'Ham',
                            hovertemplate: '<b>Ham</b><br>Prediction #%{x}<br>Confidence: %{y:.2%}<extra></extra>',
                        },
                    ]}
                    layout={{
                        width: 920,
                        height: 550,
                        title: { 
                            text: 'Model Confidence Over Time',
                            font: {
                                size: 18,
                                weight: 'bold'
                            }
                        },
                        xaxis: { 
                            title: 'Prediction Number',
                            gridcolor: '#e5e5e5',
                            showgrid: true,
                        },
                        yaxis: { 
                            title: 'Confidence Score',
                            range: [0.5, 1.05],
                            tickformat: '.0%',
                            gridcolor: '#e5e5e5',
                            showgrid: true,
                        },
                        hovermode: 'closest',
                        plot_bgcolor: '#fafafa',
                        paper_bgcolor: 'white',
                        legend: {
                            x: 1,
                            xanchor: 'right',
                            y: 1,
                            bgcolor: 'rgba(255,255,255,0.8)',
                            bordercolor: '#999',
                            borderwidth: 1
                        },
                        margin: {
                            l: 80,
                            r: 40,
                            t: 80,
                            b: 80
                        }
                    }}
                    config={{
                        displayModeBar: true,
                        displaylogo: false,
                        modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
                        toImageButtonOptions: {
                            format: 'png',
                            filename: 'confidence-timeline',
                            height: 550,
                            width: 920,
                            scale: 2
                        }
                    }}
                />
            </div>
        </div>
    );
}