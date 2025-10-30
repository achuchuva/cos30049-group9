import React, { useState } from 'react';
import Tabs from '../components/Tabs';

export default function DataVisualisation() {
  const [tab, setTab] = useState('PLOTLY');
  return (
    <div>
      <section className="hero">
        <h1 className="page-title">Data Visualisation</h1>
        <Tabs tabs={['PLOTLY','CHART','D3']} current={tab} onChange={setTab} />
      </section>

      <div className="container">
        {tab === 'PLOTLY' && (
          <>
            <p>Plotly.js implementation here</p>
          </>
        )}
        {tab === 'CHART' && <p>Chart.js implementation here</p>}
        {tab === 'D3' && <p>D3.js implementation here</p>}
      </div>
    </div>
  );
}
