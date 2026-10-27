import React, { useState, useEffect } from 'react';
import Header from '../components/Header';
import { getLatestEvaluation, getMisclassifiedExamples } from '../services/api';
import { ChevronLeft, ChevronRight, BarChart3, AlertOctagon, Cpu, Zap } from 'lucide-react';

const EvaluationPage = () => {
  const [evaluation, setEvaluation] = useState(null);
  const [errors, setErrors] = useState({ items: [], total: 0, page: 1 });
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(true);

  const fetchEvalData = async () => {
    try {
      setLoading(true);
      const evalRes = await getLatestEvaluation();
      setEvaluation(evalRes);
      const errRes = await getMisclassifiedExamples(page, 10);
      setErrors(errRes);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchEvalData();
  }, [page]);

  const metrics = evaluation?.metrics;
  const labels = metrics?.labels || ['World', 'Sports', 'Business', 'Sci/Tech'];
  const cm = metrics?.confusion_matrix || [];

  return (
    <div className="flex-1 flex flex-col overflow-y-auto bg-slate-950">
      <Header 
        title="Official Test Set Evaluation" 
        description="Comprehensive evaluation report computed on the untouched 7,600 AG News test split."
      />

      <main className="p-8 space-y-8">
        {!metrics ? (
          <div className="p-12 rounded-2xl bg-slate-900/80 border border-slate-800 text-center space-y-3">
            <BarChart3 className="w-10 h-10 text-slate-500 mx-auto" />
            <h3 className="text-base font-bold text-white">No Evaluation Results Available</h3>
            <p className="text-xs text-slate-400 max-w-md mx-auto">
              Please train a model from the Training Console. Once a model is trained, the official test set is automatically evaluated and detailed metrics will appear here.
            </p>
          </div>
        ) : (
          <>
            {/* Summary Highlights */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800">
                <span className="text-xs text-slate-400 block mb-1">Test Accuracy</span>
                <span className="text-3xl font-extrabold text-emerald-400">
                  {(metrics.accuracy * 100).toFixed(1)}%
                </span>
                <span className="text-[11px] text-slate-500 block mt-1">Total: {metrics.total_samples} examples</span>
              </div>

              <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800">
                <span className="text-xs text-slate-400 block mb-1">Macro F1-Score</span>
                <span className="text-3xl font-extrabold text-cyan-400">
                  {(metrics.macro_f1 * 100).toFixed(1)}%
                </span>
                <span className="text-[11px] text-slate-500 block mt-1">Unweighted category mean</span>
              </div>

              <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800">
                <span className="text-xs text-slate-400 block mb-1">Inference Latency</span>
                <span className="text-3xl font-extrabold text-amber-400">
                  {metrics.inference_latency_ms} <span className="text-sm font-normal text-slate-400">ms</span>
                </span>
                <span className="text-[11px] text-slate-500 block mt-1">Average per sample</span>
              </div>

              <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800">
                <span className="text-xs text-slate-400 block mb-1">Trainable Parameters</span>
                <span className="text-3xl font-extrabold text-indigo-400">
                  {metrics.param_count ? metrics.param_count.toLocaleString() : '--'}
                </span>
                <span className="text-[11px] text-slate-500 block mt-1">Model complexity</span>
              </div>
            </div>

            {/* Per-Class Classification Report Table */}
            <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-4">
              <h3 className="text-sm font-semibold text-white">Per-Class Performance Report</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead>
                    <tr className="border-b border-slate-800 text-slate-400 font-medium">
                      <th className="pb-3">Class Category</th>
                      <th className="pb-3">Precision</th>
                      <th className="pb-3">Recall</th>
                      <th className="pb-3">F1-Score</th>
                      <th className="pb-3">Support</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-850 text-slate-300">
                    {metrics.per_class && Object.entries(metrics.per_class).map(([cat, val]) => (
                      <tr key={cat} className="hover:bg-slate-850/40">
                        <td className="py-3 font-semibold text-white">{cat}</td>
                        <td className="py-3">{(val.precision * 100).toFixed(1)}%</td>
                        <td className="py-3">{(val.recall * 100).toFixed(1)}%</td>
                        <td className="py-3 font-semibold text-cyan-400">{(val.f1_score * 100).toFixed(1)}%</td>
                        <td className="py-3 text-slate-400">{val.support}</td>
                      </tr>
                    ))}
                    <tr className="font-semibold text-white pt-2 bg-slate-950/40">
                      <td className="py-3">Macro Average</td>
                      <td className="py-3">{(metrics.macro_precision * 100).toFixed(1)}%</td>
                      <td className="py-3">{(metrics.macro_recall * 100).toFixed(1)}%</td>
                      <td className="py-3 text-cyan-400">{(metrics.macro_f1 * 100).toFixed(1)}%</td>
                      <td className="py-3 text-slate-400">{metrics.total_samples}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            {/* Confusion Matrix Heatmap */}
            <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-4">
              <div>
                <h3 className="text-sm font-semibold text-white">Confusion Matrix (4 Classes)</h3>
                <p className="text-xs text-slate-400">Rows represent Actual categories, columns represent Predicted categories</p>
              </div>

              <div className="overflow-x-auto pt-2">
                <table className="border-collapse text-xs mx-auto">
                  <thead>
                    <tr>
                      <th className="p-2"></th>
                      {labels.map((l) => (
                        <th key={l} className="p-3 text-center text-slate-400 font-semibold w-24">
                          Pred: {l}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {cm.map((row, rIdx) => (
                      <tr key={rIdx}>
                        <th className="p-3 text-right text-slate-400 font-semibold whitespace-nowrap">
                          Act: {labels[rIdx]}
                        </th>
                        {row.map((cell, cIdx) => {
                          const isDiagonal = rIdx === cIdx;
                          return (
                            <td
                              key={cIdx}
                              className={"p-4 text-center font-bold text-sm border border-slate-800 " + (
                                isDiagonal
                                  ? 'bg-cyan-500/20 text-cyan-300'
                                  : cell > 0 ? 'bg-slate-800/60 text-slate-400' : 'bg-slate-950 text-slate-600'
                              )}
                            >
                              {cell}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Error Analysis & Misclassified Samples */}
            <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-sm font-semibold text-white">Error Analysis (Misclassified Test Examples)</h3>
                  <p className="text-xs text-slate-400">Inspection of false positives and negatives on the test split</p>
                </div>
                <div className="flex items-center space-x-2 text-xs">
                  <button
                    onClick={() => setPage(p => Math.max(1, p - 1))}
                    disabled={page === 1}
                    className="p-1.5 rounded-lg bg-slate-800 text-slate-300 hover:text-white disabled:opacity-40"
                  >
                    <ChevronLeft className="w-4 h-4" />
                  </button>
                  <span className="text-slate-400">Page {page} of {Math.ceil((errors.total || 1) / 10)}</span>
                  <button
                    onClick={() => setPage(p => p + 1)}
                    disabled={page >= Math.ceil((errors.total || 1) / 10)}
                    className="p-1.5 rounded-lg bg-slate-800 text-slate-300 hover:text-white disabled:opacity-40"
                  >
                    <ChevronRight className="w-4 h-4" />
                  </button>
                </div>
              </div>

              <div className="space-y-3">
                {errors.items.map((err) => (
                  <div key={err.id} className="p-4 rounded-xl bg-slate-950 border border-slate-800 text-xs space-y-2">
                    <p className="text-slate-300 leading-relaxed">{err.text}</p>
                    <div className="flex items-center space-x-3 text-[11px] pt-1">
                      <span className="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                        Actual: <strong>{err.actual_category}</strong>
                      </span>
                      <span className="px-2 py-0.5 rounded bg-rose-500/10 text-rose-400 border border-rose-500/20">
                        Predicted: <strong>{err.predicted_category}</strong>
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}
      </main>
    </div>
  );
};

export default EvaluationPage;
