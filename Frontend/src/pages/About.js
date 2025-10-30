import React, { useState } from 'react';
import Tabs from '../components/Tabs';

export default function About() {
  const [tab, setTab] = useState('OVERVIEW');
  return (
    <div>
      <section className="hero">
        <h1 className="page-title">About this project</h1>
        <p className="team-lead">Team members:</p>
        <p className="team">Anton, Josh, Kim</p>
        <Tabs tabs={['OVERVIEW','SPECIFICATIONS','CONSIDERATIONS']} current={tab} onChange={setTab} />
      </section>

      <div className="container">
        {tab === 'OVERVIEW' && (
          <>
            <p><strong>Project Overview:</strong> Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua...</p>
            <div className="image-placeholder">
              <span role="img" aria-label="placeholder">🖼️</span>
            </div>
          </>
        )}
        {tab === 'SPECIFICATIONS' && <p>Specifications content goes here.</p>}
        {tab === 'CONSIDERATIONS' && <p>Considerations content goes here.</p>}
      </div>
    </div>
  );
}
