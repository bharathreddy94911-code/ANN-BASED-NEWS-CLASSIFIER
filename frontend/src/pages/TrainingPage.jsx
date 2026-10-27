import React, { useState, useEffect } from 'react';
import Header from '../components/Header';
import { startTraining, getTrainingStatus } from '../services/api';
import { Play, Activity, CheckCircle, AlertTriangle, Cpu } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend } from 'recharts';

const TrainingPage = () => {
  const [params, setParams] = useState({
    model_type: 'cbow_mlp',
    optimizer: 'sgd',
    epochs: 5,
    batch_size: 64,
    lr: 0.1,
    embed_dim: 100,
    hidden_dim: 128,
    dropout: 0.2,
    cbow_window: 2,
    num_layers: 1,
    quick_mode: true,
    subset_size: 10000,
  });

  const [jobId, setJobId] = useState(null);
  const [jobStatus, setJobStatus] = useState(null);
  const [error, setError] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleStart = async (e) => {
    e.preventDefault();
    try {
      setIsSubmitting(true);
      setError(null);
      const res = await startTraining(params);
      setJobId(res.job_id);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to start training.');
    } finally {
      setIsSubmitting(false);
    }
  };

  useEffect(() => {
    if (!jobId) return;

    const interval = setInterval(async () => {
      try {
        const status = await getTrainingStatus(jobId);
        setJobStatus(status);
        if (status.status === 'completed' || status.status === 'failed') {
          clearInterval(interval);
        }
      } catch (err) {
        console.error(err);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [jobId]);

  return (
    <div className="flex-1 flex flex-col overflow-y-auto bg-slate-950">
      <Header 
        title="Model Training Console" 
        description="Configure hyperparameters, select optimizers (Mini-batch SGD vs Adam), and monitor real-time epoch metrics."
      />

      <main className="p-8 max-w-6xl space-y-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Training Configuration Form */}
          <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-6">
            <h3 className="text-sm font-semibold text-white tracking-wide uppercase">Hyperparameters</h3>

            <form onSubmit={handleStart} className="space-y-4 text-xs">
              <div>
                <label className="text-slate-400 block mb-1">Architecture</label>
                <select
                  value={params.model_type}
                  onChange={(e) => setParams({ ...params, model_type: e.target.value })}
                  className="w-full p-2.5 rounded-xl bg-slate-950 border border-slate-800 text-slate-200 focus:outline-none focus:border-cyan-500"
                >
                  <option value="cbow_mlp">CBOW Embeddings + MLP (Proposed)</option>
                  <option value="baseline">TF-IDF + MLP (Baseline)</option>
                </select>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-slate-400 block mb-1">Optimizer</label>
                  <select
                    value={params.optimizer}
                    onChange={(e) => setParams({ ...params, optimizer: e.target.value })}
                    className="w-full p-2.5 rounded-xl bg-slate-950 border border-slate-800 text-slate-200 focus:outline-none focus:border-cyan-500 uppercase"
                  >
                    <option value="sgd">SGD (Mini-Batch)</option>
                    <option value="adam">Adam</option>
                  </select>
                </div>
                <div>
                  <label className="text-slate-400 block mb-1">Learning Rate</label>
                  <input
                    type="number"
                    step="0.001"
                    value={params.lr}
                    onChange={(e) => setParams({ ...params, lr: parseFloat(e.target.value) })}
                    className="w-full p-2.5 rounded-xl bg-slate-950 border border-slate-800 text-slate-200 focus:outline-none focus:border-cyan-500"
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-slate-400 block mb-1">Epochs</label>
                  <input
                    type="number"
                    min="1"
                    max="50"
                    value={params.epochs}
                    onChange={(e) => setParams({ ...params, epochs: parseInt(e.target.value) })}
                    className="w-full p-2.5 rounded-xl bg-slate-950 border border-slate-800 text-slate-200 focus:outline-none focus:border-cyan-500"
                  />
                </div>
                <div>
                  <label className="text-slate-400 block mb-1">Batch Size</label>
                  <select
                    value={params.batch_size}
                    onChange={(e) => setParams({ ...params, batch_size: parseInt(e.target.value) })}
                    className="w-full p-2.5 rounded-xl bg-slate-950 border border-slate-800 text-slate-200 focus:outline-none focus:border-cyan-500"
                  >
                    <option value={32}>32</option>
                    <option value={64}>64</option>
                    <option value={128}>128</option>
                    <option value={256}>256</option>
                  </select>
                </div>
              </div>

              {params.model_type === 'cbow_mlp' && (
                <>
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="text-slate-400 block mb-1">Embedding Dim</label>
                      <input
                        type="number"
                        value={params.embed_dim}
                        onChange={(e) => setParams({ ...params, embed_dim: parseInt(e.target.value) })}
                        className="w-full p-2.5 rounded-xl bg-slate-950 border border-slate-800 text-slate-200 focus:outline-none focus:border-cyan-500"
                      />
                    </div>
                    <div>
                      <label className="text-slate-400 block mb-1">CBOW Window</label>
                      <input
                        type="number"
                        min="1"
                        max="5"
                        value={params.cbow_window}
                        onChange={(e) => setParams({ ...params, cbow_window: parseInt(e.target.value) })}
                        className="w-full p-2.5 rounded-xl bg-slate-950 border border-slate-800 text-slate-200 focus:outline-none focus:border-cyan-500"
                      />
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="text-slate-400 block mb-1">Hidden Units</label>
                      <input
                        type="number"
                        value={params.hidden_dim}
                        onChange={(e) => setParams({ ...params, hidden_dim: parseInt(e.target.value) })}
                        className="w-full p-2.5 rounded-xl bg-slate-950 border border-slate-800 text-slate-200 focus:outline-none focus:border-cyan-500"
                      />
                    </div>
                    <div>
                      <label className="text-slate-400 block mb-1">Dropout</label>
                      <input
                        type="number"
                        step="0.05"
                        min="0"
                        max="0.8"
                        value={params.dropout}
                        onChange={(e) => setParams({ ...params, dropout: parseFloat(e.target.value) })}
                        className="w-full p-2.5 rounded-xl bg-slate-950 border border-slate-800 text-slate-200 focus:outline-none focus:border-cyan-500"
                      />
                    </div>
                  </div>

                  <div>
                    <label className="text-slate-400 block mb-1">Pooling Strategy</label>
                    <select
                      value={params.pooling || 'concat'}
                      onChange={(e) => setParams({ ...params, pooling: e.target.value })}
                      className="w-full p-2.5 rounded-xl bg-slate-950 border border-slate-800 text-slate-200 focus:outline-none focus:border-cyan-500"
                    >
                      <option value="concat">Concatenated (Mean + Max) - Best</option>
                      <option value="mean">Masked Mean Pooling</option>
                      <option value="max">Masked Max Pooling</option>
                    </select>
                  </div>
                </>
              )}

              {/* Quick Debug Mode Toggle */}
              <div className="p-3.5 rounded-xl bg-slate-950/70 border border-slate-800 flex items-center justify-between">
                <div>
                  <span className="text-white font-semibold block">Quick Debug Mode</span>
                  <span className="text-slate-500 text-[11px]">Subset of 5,000 records for fast training</span>
                </div>
                <input
                  type="checkbox"
                  checked={params.quick_mode}
                  onChange={(e) => setParams({ ...params, quick_mode: e.target.checked })}
                  className="w-4 h-4 rounded text-cyan-500 bg-slate-900 border-slate-700"
                />
              </div>

              {error && (
                <div className="p-3 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-400 text-xs flex items-center space-x-2">
                  <AlertTriangle className="w-4 h-4 shrink-0" />
                  <span>{error}</span>
                </div>
              )}

              <button
                type="submit"
                disabled={isSubmitting || jobStatus?.status === 'running'}
                className="w-full py-3 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-xs transition-colors flex items-center justify-center space-x-2 disabled:opacity-40"
              >
                <Play className="w-4 h-4" />
                <span>Start Training Job</span>
              </button>
            </form>
          </div>

          {/* Training Monitor & Status */}
          <div className="lg:col-span-2 space-y-6">
            <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-6">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-sm font-semibold text-white">Live Execution Progress</h3>
                  <p className="text-xs text-slate-400">
                    {jobStatus ? ('Job ID: ' + jobStatus.job_id) : 'No active or recent job'}
                  </p>
                </div>
                {jobStatus && (
                  <span className={'px-2.5 py-1 rounded-full text-xs font-semibold uppercase ' + (
                    jobStatus.status === 'completed' ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' :
                    jobStatus.status === 'running' ? 'bg-amber-500/10 text-amber-400 border border-amber-500/20 animate-pulse' :
                    'bg-slate-800 text-slate-400'
                  )}>
                    {jobStatus.status}
                  </span>
                )}
              </div>

              {jobStatus ? (
                <div className="space-y-4">
                  <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 text-xs space-y-2">
                    <div className="flex justify-between text-slate-400">
                      <span>Status Message:</span>
                      <span className="text-white font-medium">{jobStatus.message}</span>
                    </div>
                    <div className="flex justify-between text-slate-400">
                      <span>Epoch:</span>
                      <span className="text-white font-medium">{jobStatus.current_epoch} / {jobStatus.total_epochs}</span>
                    </div>
                    <div className="flex justify-between text-slate-400">
                      <span>Elapsed Time:</span>
                      <span className="text-white font-medium">{jobStatus.elapsed_seconds} s</span>
                    </div>
                    {jobStatus.val_accuracy !== null && (
                      <div className="flex justify-between text-slate-400">
                        <span>Validation Accuracy:</span>
                        <span className="text-emerald-400 font-bold">{(jobStatus.val_accuracy * 100).toFixed(1)}%</span>
                      </div>
                    )}
                  </div>

                  {/* Loss Curves */}
                  {jobStatus.history && jobStatus.history.length > 0 && (
                    <div className="space-y-2">
                      <h4 className="text-xs font-semibold text-slate-300">Loss Curves per Epoch</h4>
                      <div className="h-56 w-full bg-slate-950 p-2 rounded-xl border border-slate-800">
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={jobStatus.history}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                            <XAxis dataKey="epoch" stroke="#64748b" fontSize={11} />
                            <YAxis stroke="#64748b" fontSize={11} />
                            <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '11px' }} />
                            <Legend wrapperStyle={{ fontSize: '11px' }} />
                            <Line type="monotone" dataKey="train_loss" name="Train Loss" stroke="#38bdf8" strokeWidth={2} dot={{ r: 3 }} />
                            <Line type="monotone" dataKey="val_loss" name="Val Loss" stroke="#f43f5e" strokeWidth={2} dot={{ r: 3 }} />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="py-16 text-center text-xs text-slate-500">
                  Select parameters on the left and click "Start Training Job" to begin.
                </div>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default TrainingPage;
