import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import './App.css';
import LandingPage from './Components/LandingPage/LandingPage';
import MazsAI from './Components/MazsAI/MazsAI_UI';
function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/MazsAI" element={<MazsAI />} />
      </Routes>
    </Router>
  );
}

export default App;
