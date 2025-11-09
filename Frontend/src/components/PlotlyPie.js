import React from 'react';
import Plot from 'react-plotly.js';
import axios from 'axios'

import { useEffect, useState } from 'react';

const API_BASE = 'http://localhost:8000';

function PlotlyPie() {
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
        <section>
            <Plot
                data={[
                    { 
                        type: 'pie', 
                        x: ["Spam", "Ham"], 
                        y: [stats.spam_count, stats.ham_count],
                        textinfo: "label+percent",
                        insidetextfont: { color: "#e5e7eb" }
                    },
                ]}
                layout={{ 
                    width: 1000, 
                    height: 750, 
                    title: { text: 'Pie chart prediction statistics' },
                    color: "#e5e7eb",
                    paper_bgcolor:'rgba(0,0,0,0)',
                    plot_bgcolor:'rgba(0,0,0,0)',
                }}
            />
        </section>
    );
}

export default PlotlyPie;