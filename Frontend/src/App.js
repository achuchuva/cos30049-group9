import React from 'react';
import { Routes, Route, NavLink, Link } from 'react-router-dom';
import About from './pages/About';
import DataVisualisation from './pages/DataVisualisation';
import Detect from './pages/Detect';
import './index.css';
import SpamDemo from './SpamDemo';

export default function App() {
  // return SpamDemo();
  return (
    <div className="app-shell">
      <header className="topbar">
        <Link to="/" className="brand" style={{ textDecoration: 'none', color: 'inherit' }}>
          <span className="home-icon">🏠</span>
          <span className="brand-text">COS30049 Group 9 Project</span>
        </Link>

        <nav className="navlinks">
          <NavLink to="/visualisation" className={({isActive}) => isActive ? 'navlink active' : 'navlink'}>
            Data Visualisation
          </NavLink>
          <NavLink to="/about" className={({isActive}) => isActive ? 'navlink active' : 'navlink'}>
            About
          </NavLink>
        </nav>
      </header>

      <main>
        <Routes>
          <Route path="/" element={<Detect />} />
          <Route path="/visualisation" element={<DataVisualisation />} />
          <Route path="/about" element={<About />} />
        </Routes>
      </main>
    </div>
  );
}
