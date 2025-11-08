import React, { useState, useEffect, useCallback } from 'react';
import { Routes, Route, NavLink, Link, useLocation } from 'react-router-dom';
import About from './pages/About';
import Results from './pages/Results';
import Detect from './pages/Detect';
import Emails from './pages/Emails';
import './index.css';

const API_BASE = 'http://localhost:8000';
const LAST_SEEN_TIMESTAMP_KEY = 'lastSeenEmailTimestamp';

export default function App() {
    const [unreadCount, setUnreadCount] = useState(0);
    const location = useLocation();

    const fetchUnreadCount = useCallback(async () => {
        try {
            const res = await fetch(`${API_BASE}/api/v1/email/predictions?limit=50`);
            if (res.ok) {
                const data = await res.json();
                const predictions = data.predictions || [];
                const lastSeenTimestamp = localStorage.getItem(LAST_SEEN_TIMESTAMP_KEY);

                if (lastSeenTimestamp) {
                    const newEmails = predictions.filter(p => new Date(p.timestamp) > new Date(lastSeenTimestamp));
                    setUnreadCount(newEmails.length);
                } else {
                    setUnreadCount(predictions.length);
                }
            }
        } catch (err) {
            console.error('Failed to fetch unread count:', err);
        }
    }, []);

    useEffect(() => {
        fetchUnreadCount();
        const interval = setInterval(fetchUnreadCount, 10000); // Poll every 10 seconds
        return () => clearInterval(interval);
    }, [fetchUnreadCount]);

    useEffect(() => {
        const handlePageVisit = async () => {
            if (location.pathname === '/emails') {
                try {
                    const res = await fetch(`${API_BASE}/api/v1/email/predictions?limit=1`);
                    if (res.ok) {
                        const data = await res.json();
                        if (data.predictions && data.predictions.length > 0) {
                            const latestTimestamp = data.predictions[0].timestamp;
                            localStorage.setItem(LAST_SEEN_TIMESTAMP_KEY, latestTimestamp);
                        }
                        setUnreadCount(0);
                    }
                } catch (err) {
                    console.error('Failed to update last seen timestamp:', err);
                }
            }
        };
        handlePageVisit();
    }, [location]);

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
                    <NavLink to="/emails" className={({ isActive }) => isActive ? 'navlink active' : 'navlink'}>
                        Emails
                        {unreadCount > 0 && <span className="notification-bubble">{unreadCount}</span>}
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
                    <Route path="/emails" element={<Emails />} />
                </Routes>
            </main>
        </div>
    );
}
