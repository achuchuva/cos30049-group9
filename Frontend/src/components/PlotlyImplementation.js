import React from 'react';
import Plot from 'react-plotly.js';
import axios from 'axios'

import { useEffect, useState } from 'react';

const API_BASE = 'http://localhost:8000';

export default function PlotlyImplementation() {
    const [stats, setStats] = useState([null]);
    const [error, setError] = useState(null);

    useEffect(()=>{
        axios.get(`${API_BASE}/api/v1/stats`)
            .then((response)=>{
                if(response.status===200){   
                    console.log(response.data)
                    setStats(response.data)
                }
                else{
                    setError("ERROR: "+response.status+" , "+response.statusText)
                }
            })
            .catch((err) => console.log.err);
    },[]);
    
    return (
        <Plot
            data={[
                {
                    x: [1, 2, 3],
                    y: [2, 6, 3],
                    type: 'scatter',
                    mode: 'lines+markers',
                    marker: { color: 'red' },
                },
                { type: 'bar', x: [1, 2, 3], y: [stats.total_predictions, stats.spam_count, stats.ham_count] },
            ]}
            layout={{ width: 1000, height: 750, title: { text: 'A Fancy Plot' } }}
        />
    );
}