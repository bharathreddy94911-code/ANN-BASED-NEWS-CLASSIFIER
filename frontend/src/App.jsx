import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import DatasetPage from './pages/DatasetPage';
import ClassifierPage from './pages/ClassifierPage';
import TrainingPage from './pages/TrainingPage';
import ExperimentsPage from './pages/ExperimentsPage';
import EvaluationPage from './pages/EvaluationPage';
import AboutPage from './pages/AboutPage';

function App() {
  return (
    <Router>
      <div className="flex h-screen w-screen overflow-hidden bg-slate-950 text-slate-100">
        <Sidebar />
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/dataset" element={<DatasetPage />} />
          <Route path="/classifier" element={<ClassifierPage />} />
          <Route path="/training" element={<TrainingPage />} />
          <Route path="/experiments" element={<ExperimentsPage />} />
          <Route path="/evaluation" element={<EvaluationPage />} />
          <Route path="/about" element={<AboutPage />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
