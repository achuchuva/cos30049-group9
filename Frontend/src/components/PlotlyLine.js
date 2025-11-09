import React from 'react';
import Plot from 'react-plotly.js';
import axios from 'axios'

import { useEffect, useState } from 'react';
import { forEach } from 'jszip';

const API_BASE = 'http://localhost:8000';

function PlotlyLine() {
    const [stats, setStats] = useState([null]);
    const [error, setError] = useState(null);

    var spamtrace = {
        x: [],
        y: [],
        mode: 'lines+markers',
        name: 'Spam',
        marker: {
            color: 'rgba(231, 54, 0, 1)',
            size: 8
        },
        line: {
            color: 'rgba(148, 22, 0, 1)',
            width: 1
        }
    };

    var hamtrace = {
        x: [],
        y: [],
        mode: 'lines+markers',
        name: 'Ham',
        marker: {
            color: 'rgba(0, 231, 58, 1)',
            size: 8
        },
        line: {
            color: 'rgba(0, 148, 32, 1)',
            width: 1
        }
    };

    const appendTraces = stats => {
        for (let idx in stats.confidence_distribution){
            const item = stats.confidence_distribution[idx];
            if (item.prediction == "spam"){
                spamtrace.y.push(item.confidence);
                spamtrace.x.push(idx);
            }
            else{
                hamtrace.y.push(item.confidence);
                hamtrace.x.push(idx);
            }
        }
    };

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
        <section>
            {appendTraces(stats)}
            <Plot
                data={[spamtrace, hamtrace]}
                layout={{ 
                    width: 1000, 
                    height: 750, 
                    title: { text: 'Line+scatter chart prediction statistics' },
                }}
            />
        </section>
    );
}

export default PlotlyLine;