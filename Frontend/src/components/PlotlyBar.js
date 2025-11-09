import React from 'react';
import Plot from 'react-plotly.js';
import axios from 'axios'

import { useEffect, useState } from 'react';

const API_BASE = 'http://localhost:8000';

function PlotlyBar() {
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
                        type: 'bar', 
                        x: ["Total prediction", "Spam", "Ham"], 
                        y: [stats.total_predictions, stats.spam_count, stats.ham_count] ,
                        marker: {color: ['gray', 'red', 'green'] },
                    },
                ]}
                layout={{ 
                    width: 1000, 
                    height: 750, 
                    title: { text: 'Bar chart prediction statistics' } 
                }}
            />
        </section>
    );
}


export default PlotlyBar;