import React, { useState } from 'react';
import Tabs from '../components/Tabs';
import Chartjs from '../components/ChartjsImplementation';
import Plotly from '../components/PlotlyImplementation';

export default function DataVisualisation() {
  const [tab, setTab] = useState('PLOTLY');

  return (
    <div>
      <section className="hero">
        <h1 className="page-title">Data Visualisation</h1>
        <Tabs tabs={['PLOTLY', 'CHART', 'D3']} current={tab} onChange={setTab} />
      </section>

      <div className="container">
        {tab === 'PLOTLY' && (
          <Plotly />
        )}
        {tab === 'CHART' && (
          <Chartjs />
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
