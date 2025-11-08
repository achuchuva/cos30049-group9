import React from 'react';
import { Routes, Route, NavLink, Link } from 'react-router-dom';
import About from './pages/About';
import Results from './pages/Results';
import Detect from './pages/Detect';
import './index.css';

export default function App() {
    return (
        <div className="app-shell">
            <header className="topbar">
                <Link to="/" className="brand" style={{ textDecoration: 'none', color: 'inherit' }}>
                    <span className="home-icon">🏠</span>
                    <span className="brand-text">COS30049 Assignment 3</span>
                </Link>

                <nav className="navlinks">
                    <NavLink to="/results" className={({ isActive }) => isActive ? 'navlink active' : 'navlink'}>
                        Results
                    </NavLink>
                    <NavLink to="/about" className={({ isActive }) => isActive ? 'navlink active' : 'navlink'}>
                        About
                    </NavLink>
                </nav>
            </header>

            <main>
                <Routes>
                    <Route path="/" element={<Detect />} />
                    <Route path="/results" element={<Results />} />
                    <Route path="/about" element={<About />} />
                </Routes>
            </main>
        </div>
    );
}
