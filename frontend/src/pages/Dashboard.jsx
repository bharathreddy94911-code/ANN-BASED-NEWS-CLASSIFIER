import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import Header from '../components/Header';
import { getHealth, getDatasetInfo, getExperiments, getLatestEvaluation } from '../services/api';
import { 
  Database, 
  BrainCircuit, 
  Layers, 
  Target, 
  Sparkles, 
  CheckCircle2, 
  AlertCircle,
  ArrowRight,
  Clock,
  TrendingUp
} from 'lucide-react';

const Dashboard = () => {
  const [health, setHealth] = useState(null);
  const [datasetInfo, setDatasetInfo] = useState(null);
  const [experiments, setExperiments] = useState([]);
  const [latestEval, setLatestEval] = useState(null);
  const [loading, setLoading] = useState(true);

  const fetchData = async () => {
    try {
      setLoading(true);
      const [hRes, dsRes, expRes] = await Promise.all([
        getHealth().catch(() => null),
        getDatasetInfo().catch(() => null),
        getExperiments().catch(() => []),
      ]);
      setHealth(hRes);
      setDatasetInfo(dsRes);
      setExperiments(expRes || []);

      try {
        const evalRes = await getLatestEvaluation();
        setLatestEval(evalRes);
      } catch {
        setLatestEval(null);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  return (
    <div className="flex-1 flex flex-col overflow-y-auto bg-slate-950">
      <Header 
        title="Project Overview & Metrics Dashboard" 
        description="Continuous Bag of Words (CBOW) embeddings with Multi-Layer Perceptron (MLP) classification."
        isBusy={health?.is_training}
        onRefresh={fetchData}
      />

      <main className="p-8 space-y-8">
        {/* Metric Cards Row */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
          <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800/80">
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs font-medium text-slate-400">Dataset Records</span>
              <Database className="w-4 h-4 text-cyan-400" />
            </div>
            <div className="text-2xl font-bold text-white tracking-tight">
              {datasetInfo ? datasetInfo.train_records.toLocaleString() : '120,000'}
            </div>
            <p className="text-xs text-slate-500 mt-1">
              Train: {datasetInfo ? datasetInfo.train_records.toLocaleString() : '120,000'} | Test: {datasetInfo ? datasetInfo.test_records.toLocaleString() : '7,600'}
            </p>
          </div>

          <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800/80">
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs font-medium text-slate-400">Model Status</span>
              <BrainCircuit className="w-4 h-4 text-indigo-400" />
            </div>
            <div className="flex items-center space-x-2">
              <span className="text-2xl font-bold text-white tracking-tight">
                {health?.model_available ? 'Active' : 'Not Trained'}
              </span>
              {health?.model_available ? (
                <CheckCircle2 className="w-5 h-5 text-emerald-400" />
              ) : (
                <AlertCircle className="w-5 h-5 text-amber-400" />
              )}
            </div>
            <p className="text-xs text-slate-500 mt-1">
              {health?.model_info?.architecture || 'CBOW + MLP Classifier'}
            </p>
          </div>

          <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800/80">
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs font-medium text-slate-400">Latest Test Accuracy</span>
              <Target className="w-4 h-4 text-emerald-400" />
            </div>
            <div className="text-2xl font-bold text-white tracking-tight">
              {latestEval ? (latestEval.metrics.accuracy * 100).toFixed(1) + '%' : '--'}
            </div>
            <p className="text-xs text-slate-500 mt-1">
              {latestEval ? 'Macro F1: ' + (latestEval.metrics.macro_f1 * 100).toFixed(1) + '%' : 'Requires test evaluation'}
            </p>
          </div>

          <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800/80">
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs font-medium text-slate-400">Experiments Logged</span>
              <Layers className="w-4 h-4 text-amber-400" />
            </div>
            <div className="text-2xl font-bold text-white tracking-tight">
              {experiments.length}
            </div>
            <p className="text-xs text-slate-500 mt-1">Controlled runs recorded</p>
          </div>
        </div>

        {/* Quick Actions / Call to Action */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="p-6 rounded-2xl bg-gradient-to-br from-slate-900 via-slate-900 to-indigo-950/40 border border-slate-800 flex flex-col justify-between">
            <div>
              <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 mb-3">
                Live Prediction
              </span>
              <h3 className="text-lg font-semibold text-white mb-1">Classify Real-Time News</h3>
              <p className="text-xs text-slate-400 leading-relaxed mb-4">
                Input any news headline or story snippet to inspect the predicted category (World, Sports, Business, Sci/Tech), model confidence, and inference latency.
              </p>
            </div>
            <Link
              to="/classifier"
              className="inline-flex items-center justify-center space-x-2 px-4 py-2.5 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-semibold text-xs transition-colors self-start"
            >
              <Sparkles className="w-4 h-4" />
              <span>Open News Classifier</span>
            </Link>
          </div>

          <div className="p-6 rounded-2xl bg-gradient-to-br from-slate-900 via-slate-900 to-slate-850 border border-slate-800 flex flex-col justify-between">
            <div>
              <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 mb-3">
                Model Training
              </span>
              <h3 className="text-lg font-semibold text-white mb-1">Train & Benchmark Models</h3>
              <p className="text-xs text-slate-400 leading-relaxed mb-4">
                Train CBOW embeddings and the MLP classifier using Mini-Batch Gradient Descent (SGD) or Adam, or compare against the TF-IDF baseline.
              </p>
            </div>
            <Link
              to="/training"
              className="inline-flex items-center justify-center space-x-2 px-4 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-white font-medium text-xs transition-colors self-start border border-slate-700"
            >
              <span>Go to Training Console</span>
              <ArrowRight className="w-4 h-4" />
            </Link>
          </div>
        </div>

        {/* Recent Experiments Table */}
        <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800/80">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-sm font-semibold text-white">Recent Controlled Experiments</h3>
              <p className="text-xs text-slate-400">Comparison of model runs and parameters</p>
            </div>
            <Link to="/experiments" className="text-xs text-cyan-400 hover:underline">
              View all
            </Link>
          </div>

          {experiments.length === 0 ? (
            <div className="py-8 text-center text-xs text-slate-500">
              No experiment runs recorded yet. Start training or launch an experiment to track results.
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-slate-800 text-slate-400 font-medium">
                    <th className="pb-3">Name</th>
                    <th className="pb-3">Model</th>
                    <th className="pb-3">Optimizer</th>
                    <th className="pb-3">LR</th>
                    <th className="pb-3">Batch Size</th>
                    <th className="pb-3">Val Acc</th>
                    <th className="pb-3">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-850 text-slate-300">
                  {experiments.slice(-5).reverse().map((exp) => (
                    <tr key={exp.id} className="hover:bg-slate-850/40">
                      <td className="py-3 font-medium text-white">{exp.name}</td>
                      <td className="py-3 uppercase text-[11px] text-slate-400">{exp.config.model_type}</td>
                      <td className="py-3 uppercase text-[11px]">{exp.config.optimizer}</td>
                      <td className="py-3">{exp.config.lr}</td>
                      <td className="py-3">{exp.config.batch_size}</td>
                      <td className="py-3 font-semibold text-emerald-400">
                        {exp.results?.val_metrics?.accuracy ? (exp.results.val_metrics.accuracy * 100).toFixed(1) + '%' : '--'}
                      </td>
                      <td className="py-3">
                        <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                          {exp.status}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </main>
    </div>
  );
};

export default Dashboard;
