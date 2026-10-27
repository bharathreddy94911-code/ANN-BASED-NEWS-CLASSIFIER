import React, { useState, useEffect } from 'react';
import Header from '../components/Header';
import { getExperiments, startExperiment, getExportUrl } from '../services/api';
import { Download, FlaskConical, Play, CheckCircle2, ChevronRight } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend } from 'recharts';

const ExperimentsPage = () => {
  const [experiments, setExperiments] = useState([]);
  const [selectedSuite, setSelectedSuite] = useState('baseline_vs_cbow');
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('compare');

  const fetchExp = async () => {
    try {
      const data = await getExperiments();
      setExperiments(data || []);
    } catch (err) {
      console.error(err);
    }
  };

  useEffect(() => {
    fetchExp();
  }, []);

  const runSuite = async (suiteType) => {
    try {
      setLoading(true);
      if (suiteType === 'baseline_vs_cbow') {
        await startExperiment({
          name: 'Experiment A: TF-IDF Baseline',
          model_type: 'baseline',
          optimizer: 'sgd',
          epochs: 5,
          lr: 0.1,
          batch_size: 64,
          quick_mode: true,
          subset_size: 10000,
        });
      } else if (suiteType === 'lr_study') {
        await startExperiment({
          name: 'Experiment B: Concat Pooling (SGD lr=0.2)',
          model_type: 'cbow_mlp',
          optimizer: 'sgd',
          epochs: 6,
          lr: 0.2,
          pooling: 'concat',
          hidden_dim: 256,
          batch_size: 64,
          quick_mode: true,
          subset_size: 10000,
        });
      } else if (suiteType === 'optimizer_study') {
        await startExperiment({
          name: 'Experiment C: Concat Pooling (Adam lr=0.002)',
          model_type: 'cbow_mlp',
          optimizer: 'adam',
          epochs: 6,
          lr: 0.002,
          pooling: 'concat',
          hidden_dim: 256,
          batch_size: 64,
          quick_mode: true,
          subset_size: 10000,
        });
      }
      setTimeout(fetchExp, 1000);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const chartData = experiments.map(exp => ({
    name: exp.name.length > 20 ? exp.name.substring(0, 20) + '...' : exp.name,
    val_acc: exp.results?.val_metrics?.accuracy ? Math.round(exp.results.val_metrics.accuracy * 1000) / 10 : 0,
    test_acc: exp.results?.test_metrics?.accuracy ? Math.round(exp.results.test_metrics.accuracy * 1000) / 10 : 0,
  }));

  return (
    <div className="flex-1 flex flex-col overflow-y-auto bg-slate-950">
      <Header 
        title="Controlled Experiments Console" 
        description="Run reproducible trials across architectures, optimizers, learning rates, and export results."
      />

      <main className="p-8 space-y-8">
        {/* Preset Experiment Suites */}
        <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-4">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div>
              <h3 className="text-sm font-semibold text-white">Preset Academic Experiments</h3>
              <p className="text-xs text-slate-400">Launch standard benchmark experiments with a single click</p>
            </div>
            <a
              href={getExportUrl()}
              className="inline-flex items-center space-x-2 px-3.5 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-white font-medium text-xs transition-colors border border-slate-700 self-start"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Export CSV</span>
            </a>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2">
            <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 flex flex-col justify-between">
              <div>
                <span className="text-xs font-semibold text-cyan-400">Experiment A</span>
                <h4 className="text-sm font-bold text-white mt-0.5">Baseline vs Proposed</h4>
                <p className="text-xs text-slate-400 mt-1">TF-IDF + MLP baseline compared with CBOW + MLP.</p>
              </div>
              <button
                onClick={() => runSuite('baseline_vs_cbow')}
                disabled={loading}
                className="mt-4 px-3 py-1.5 rounded-lg bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 hover:bg-cyan-500/20 text-xs font-semibold self-start"
              >
                Run Baseline Trial
              </button>
            </div>

            <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 flex flex-col justify-between">
              <div>
                <span className="text-xs font-semibold text-indigo-400">Experiment B</span>
                <h4 className="text-sm font-bold text-white mt-0.5">Learning Rate Study</h4>
                <p className="text-xs text-slate-400 mt-1">Test learning rate scaling (0.005 vs standard 0.01).</p>
              </div>
              <button
                onClick={() => runSuite('lr_study')}
                disabled={loading}
                className="mt-4 px-3 py-1.5 rounded-lg bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 hover:bg-indigo-500/20 text-xs font-semibold self-start"
              >
                Run LR Trial
              </button>
            </div>

            <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 flex flex-col justify-between">
              <div>
                <span className="text-xs font-semibold text-emerald-400">Experiment E</span>
                <h4 className="text-sm font-bold text-white mt-0.5">Optimizer Comparison</h4>
                <p className="text-xs text-slate-400 mt-1">Compare Mini-Batch SGD against Adam optimization.</p>
              </div>
              <button
                onClick={() => runSuite('optimizer_study')}
                disabled={loading}
                className="mt-4 px-3 py-1.5 rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 hover:bg-emerald-500/20 text-xs font-semibold self-start"
              >
                Run Adam Trial
              </button>
            </div>
          </div>
        </div>

        {/* Comparison Bar Chart */}
        {chartData.length > 0 && (
          <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-4">
            <h3 className="text-sm font-semibold text-white">Accuracy Comparison Across Completed Trials</h3>
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="name" stroke="#64748b" fontSize={10} />
                  <YAxis stroke="#64748b" fontSize={11} domain={[0, 100]} />
                  <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '11px' }} />
                  <Legend wrapperStyle={{ fontSize: '11px' }} />
                  <Bar dataKey="val_acc" name="Val Accuracy (%)" fill="#38bdf8" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="test_acc" name="Test Accuracy (%)" fill="#818cf8" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* Experiment Records Table */}
        <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-4">
          <h3 className="text-sm font-semibold text-white">Logged Experiments History</h3>
          {experiments.length === 0 ? (
            <div className="py-8 text-center text-xs text-slate-500">
              No experiment trials logged yet. Click one of the buttons above to launch a trial.
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-slate-800 text-slate-400 font-medium">
                    <th className="pb-3">Experiment</th>
                    <th className="pb-3">Model</th>
                    <th className="pb-3">Optimizer</th>
                    <th className="pb-3">LR</th>
                    <th className="pb-3">Batch Size</th>
                    <th className="pb-3">Val Acc</th>
                    <th className="pb-3">Test Acc</th>
                    <th className="pb-3">Time</th>
                    <th className="pb-3">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-850 text-slate-300">
                  {experiments.map((exp) => (
                    <tr key={exp.id} className="hover:bg-slate-850/40">
                      <td className="py-3 font-semibold text-white">{exp.name}</td>
                      <td className="py-3 uppercase text-[11px] text-slate-400">{exp.config.model_type}</td>
                      <td className="py-3 uppercase text-[11px]">{exp.config.optimizer}</td>
                      <td className="py-3">{exp.config.lr}</td>
                      <td className="py-3">{exp.config.batch_size}</td>
                      <td className="py-3 font-semibold text-cyan-400">
                        {exp.results?.val_metrics?.accuracy ? (exp.results.val_metrics.accuracy * 100).toFixed(1) + '%' : '--'}
                      </td>
                      <td className="py-3 font-semibold text-emerald-400">
                        {exp.results?.test_metrics?.accuracy ? (exp.results.test_metrics.accuracy * 100).toFixed(1) + '%' : '--'}
                      </td>
                      <td className="py-3 text-slate-400">{exp.results?.training_time ? exp.results.training_time + 's' : '--'}</td>
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

export default ExperimentsPage;
