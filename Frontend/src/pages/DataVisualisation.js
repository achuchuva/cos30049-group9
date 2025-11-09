import React, { useEffect, useState } from 'react';
import Tabs from '../components/Tabs';
import ChartjsBar from '../components/ChartjsBar';
import ChartjsDoughnut from '../components/ChartjsDoughnut';
import PlotlyBar from '../components/PlotlyBar';
import PlotlyLine from '../components/PlotlyLine';
import axios from 'axios'

const API_BASE = 'http://localhost:8000';

export default function DataVisualisation() {
  const [tab, setTab] = useState('PLOTLY');
  const [health, setHealth] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [stats, setStats] = useState([null]);
  const [error, setError] = useState(null);
    
  useEffect(() => {
      const fetchHealthAndInfo = async () => {
          try {
              const healthRes = await fetch(`${API_BASE}/health`);
              if (healthRes.ok) {
                  setHealth(await healthRes.json());
              }
              const modelRes = await fetch(`${API_BASE}/api/v1/model/info`);
              if (modelRes.ok) {
                  setModelInfo(await modelRes.json());
              }
          } catch (e) {
              setError('Backend not reachable. Start FastAPI server.');
          }
      };
      fetchHealthAndInfo();
  }, []);

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
    <div>
      <section className="hero">
        <h1 className="page-title">Data Visualisation</h1>
        <Tabs tabs={['PLOTLY', 'CHART', 'D3']} current={tab} onChange={setTab} />
      </section>
      <div style={{marginTop: '0.5rem'}}>
          <div style={{display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '0.5rem', marginBottom: '1rem'}}>
              <div style={{background: '#f0f0f0', padding: '0.5rem', borderRadius: '4px'}}>
                  <p style={{margin: 0, fontSize: '0.75rem', color: '#666'}}>Total Predictions</p>
                  <p style={{margin: 0, fontSize: '1.5rem', fontWeight: 'bold'}}>{stats.total_predictions}</p>
              </div>
              <div style={{background: '#ffe5e5', padding: '0.5rem', borderRadius: '4px'}}>
                  <p style={{margin: 0, fontSize: '0.75rem', color: '#666'}}>Spam Count</p>
                  <p style={{margin: 0, fontSize: '1.5rem', fontWeight: 'bold'}}>{stats.spam_count}</p>
              </div>
              <div style={{background: '#e5f7e5', padding: '0.5rem', borderRadius: '4px'}}>
                  <p style={{margin: 0, fontSize: '0.75rem', color: '#666'}}>Ham Count</p>
                  <p style={{margin: 0, fontSize: '1.5rem', fontWeight: 'bold'}}>{stats.ham_count}</p>
              </div>
              <div style={{background: '#e5f0ff', padding: '0.5rem', borderRadius: '4px'}}>
                  <p style={{margin: 0, fontSize: '0.75rem', color: '#666'}}>Spam Rate</p>
                  <p style={{margin: 0, fontSize: '1.5rem', fontWeight: 'bold'}}>{(stats.spam_rate * 100).toFixed(1)}%</p>
              </div>
          </div>
          <div>
              <p style={{fontSize: '0.85rem', fontWeight: 'bold', marginBottom: '0.3rem'}}>Feature Averages:</p>
              <ul style={{fontSize: '0.8rem', marginTop: '0.3rem'}}>
                  <li>Avg characters: {stats.feature_averages.avg_char_count || 'N/A'}</li>
                  <li>Avg words: {stats.feature_averages.avg_word_count || 'N/A'}</li>
                  <li>Avg suspicious words: {stats.feature_averages.avg_suspicious_words || 'N/A'}</li>
                  <li>Avg URLs: {stats.feature_averages.avg_url_count || 'N/A'}</li>
              </ul>
          </div>
          <div>
              <p style={{fontSize: '0.85rem', fontWeight: 'bold', marginBottom: '0.3rem'}}>Avg Confidence: {(stats.avg_confidence * 100).toFixed(2)}%</p>
          </div>
      </div>

      <div className="container">
        {tab === 'PLOTLY' && (
          <>
            <PlotlyBar />
            <PlotlyLine />
          </>
        )}
        {tab === 'CHART' && (
          <>
            <ChartjsBar />
            <ChartjsDoughnut />
          </>
        )}
        {tab === 'D3' && (
          <>
            <p>D3.js implementation here</p>
            <div id="d3" style={{ width: 600, height: 250 }}></div>
          </>
        )}
      </div>
    </div>
  );
}
