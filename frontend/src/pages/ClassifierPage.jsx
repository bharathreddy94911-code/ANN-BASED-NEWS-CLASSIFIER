import React, { useState } from 'react';
import Header from '../components/Header';
import { predictNews } from '../services/api';
import { Sparkles, Trash2, Clock, CheckCircle, AlertCircle, HelpCircle } from 'lucide-react';

const ClassifierPage = () => {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const sampleArticles = [
    {
      title: 'World Sample',
      text: 'European leaders and delegates meet in Geneva for diplomatic talks concerning border treaties and international humanitarian support.'
    },
    {
      title: 'Sports Sample',
      text: 'Real Madrid secures dramatic extra-time win with a spectacular header in the UEFA Champions League semifinal match.'
    },
    {
      title: 'Business Sample',
      text: 'Federal Reserve holds interest rates steady as Wall Street stocks climb, with tech earnings outpacing quarterly forecasts.'
    },
    {
      title: 'Sci/Tech Sample',
      text: 'NASA scientists unveil high-resolution images from the new space telescope revealing distant galaxies and organic molecules.'
    }
  ];

  const handleClassify = async () => {
    if (!text.trim() || text.length < 5) {
      setError('Please enter at least 5 characters to classify.');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      const res = await predictNews(text);
      setResult(res);
    } catch (err) {
      setError(err.response?.data?.detail || 'Inference error. Please ensure the model is trained first.');
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  const categoryColors = {
    'World': 'text-sky-400 bg-sky-500/10 border-sky-500/20',
    'Sports': 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20',
    'Business': 'text-amber-400 bg-amber-500/10 border-amber-500/20',
    'Sci/Tech': 'text-purple-400 bg-purple-500/10 border-purple-500/20',
  };

  return (
    <div className="flex-1 flex flex-col overflow-y-auto bg-slate-950">
      <Header 
        title="Live News Classification" 
        description="Classify any custom news headline or story using the trained Continuous Bag of Words (CBOW) embeddings and MLP classifier."
      />

      <main className="p-8 max-w-5xl space-y-6">
        {/* Sample selection buttons */}
        <div>
          <span className="text-xs text-slate-400 block mb-2 font-medium">Quick Test Examples:</span>
          <div className="flex flex-wrap gap-2">
            {sampleArticles.map((sample) => (
              <button
                key={sample.title}
                onClick={() => { setText(sample.text); setError(null); }}
                className="px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 hover:border-cyan-500/40 text-slate-300 hover:text-white text-xs transition-colors"
              >
                {sample.title}
              </button>
            ))}
          </div>
        </div>

        {/* Text Input Area */}
        <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-4">
          <div className="flex items-center justify-between">
            <label className="text-xs font-semibold text-white uppercase tracking-wider">
              News Article / Headline Input
            </label>
            <span className="text-xs text-slate-500">{text.length} characters</span>
          </div>

          <textarea
            rows={5}
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Type or paste a news headline or full article text here..."
            className="w-full p-4 rounded-xl bg-slate-950 border border-slate-800 text-slate-200 placeholder-slate-600 focus:outline-none focus:border-cyan-500 text-sm leading-relaxed resize-none"
          />

          {error && (
            <div className="p-3.5 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-400 text-xs flex items-center space-x-2">
              <AlertCircle className="w-4 h-4 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          <div className="flex items-center justify-between pt-2">
            <button
              onClick={() => { setText(''); setResult(null); setError(null); }}
              className="inline-flex items-center space-x-1.5 px-3 py-2 rounded-lg text-slate-400 hover:text-slate-200 text-xs transition-colors"
            >
              <Trash2 className="w-3.5 h-3.5" />
              <span>Clear</span>
            </button>

            <button
              onClick={handleClassify}
              disabled={loading || !text.trim()}
              className="inline-flex items-center space-x-2 px-5 py-2.5 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-semibold text-xs transition-colors disabled:opacity-40"
            >
              {loading ? (
                <>
                  <div className="w-4 h-4 border-2 border-slate-950 border-t-transparent rounded-full animate-spin" />
                  <span>Classifying...</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" />
                  <span>Classify News</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Prediction Results Card */}
        {result && (
          <div className="p-6 rounded-2xl bg-slate-900/90 border border-slate-800 space-y-6">
            <div className="flex flex-col md:flex-row md:items-center justify-between pb-6 border-b border-slate-800 gap-4">
              <div>
                <span className="text-xs text-slate-400 font-medium block">Predicted Category</span>
                <div className="flex items-center space-x-3 mt-1">
                  <span className={"px-4 py-1.5 rounded-xl border text-base font-bold " + (categoryColors[result.predicted_category] || 'text-white')}>
                    {result.predicted_category}
                  </span>
                  <div className="text-xs text-slate-400">
                    Confidence: <span className="text-white font-bold">{(result.confidence * 100).toFixed(1)}%</span>
                  </div>
                </div>
              </div>

              <div className="flex items-center space-x-4 text-xs text-slate-400">
                <div className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-slate-950 border border-slate-800">
                  <Clock className="w-3.5 h-3.5 text-cyan-400" />
                  <span>Latency: <strong className="text-white">{result.latency_ms} ms</strong></span>
                </div>
                <div className="px-3 py-1.5 rounded-lg bg-slate-950 border border-slate-800">
                  <span>Version: <strong className="text-white font-mono">{result.model_version}</strong></span>
                </div>
              </div>
            </div>

            {/* Probability Breakdown */}
            <div>
              <h4 className="text-xs font-semibold text-white uppercase tracking-wider mb-4">
                Class Probability Distribution
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {Object.entries(result.probabilities).map(([cat, prob]) => (
                  <div key={cat} className="p-4 rounded-xl bg-slate-950/70 border border-slate-800 space-y-2">
                    <div className="flex items-center justify-between text-xs">
                      <span className="font-medium text-slate-300">{cat}</span>
                      <span className="font-bold text-white">{(prob * 100).toFixed(1)}%</span>
                    </div>
                    <div className="w-full h-2 rounded-full bg-slate-800 overflow-hidden">
                      <div 
                        className={"h-full rounded-full transition-all duration-500 " + (cat === result.predicted_category ? 'bg-cyan-400' : 'bg-slate-600')}
                        style={{ width: Math.max(4, prob * 100) + '%' }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <p className="text-[11px] text-slate-500 italic">
              Note: Confidence score indicates the model's Softmax output probability across the 4 news classes.
            </p>
          </div>
        )}
      </main>
    </div>
  );
};

export default ClassifierPage;
