import React, { useState } from 'react';
import Tabs from '../components/Tabs';
import Chartjs from '../components/ChartjsImplementation';


//plotly stuff
import Plot from 'react-plotly.js';

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
            <div id="plotly" style={{width:600, height:250}}></div>
            {/* <Plot data={[
              {
                x:[1,2,3],
                y:[2,6,3],
                type: 'scatter',
                marker: {color: 'red'},
              },
              {type: 'bar', x:[1,2,3], y:[2,5,3]}
            ]}
            layout={{width:320, height:240, title: {text: "test plotly"}}}
            /> */}
          </>
        )}
        {tab === 'CHART' && (
          <>
            <p>Chart.js implementation here</p>
            <div id="chartjs" style={{width:600, height:250}}></div>
            {/* Chartjs() */}
          </>
        )}
        {tab === 'D3' && (
          <>
            <p>D3.js implementation here</p>
            <div id="d3" style={{width:600, height:250}}></div>
          </>
        )}
      </div>
    </div>
  );
}
