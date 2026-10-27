import React, { useState, useEffect } from 'react';
import Header from '../components/Header';
import { getDatasetInfo, getDatasetDistribution, getDatasetExamples } from '../services/api';
import { ExternalLink, ChevronLeft, ChevronRight, Filter } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

const DatasetPage = () => {
  const [info, setInfo] = useState(null);
  const [distribution, setDistribution] = useState(null);
  const [examples, setExamples] = useState({ items: [], total: 0, page: 1 });
  const [split, setSplit] = useState('train');
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(true);

  const fetchDatasetData = async () => {
    try {
      setLoading(true);
      const [iRes, dRes, eRes] = await Promise.all([
        getDatasetInfo(),
        getDatasetDistribution(),
        getDatasetExamples(split, page, 10),
      ]);
      setInfo(iRes);
      setDistribution(dRes);
      setExamples(eRes);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDatasetData();
  }, [split, page]);

  const COLORS = ['#38bdf8', '#818cf8', '#34d399', '#f59e0b'];

  const chartData = distribution && distribution[split]
    ? Object.keys(distribution[split]).map((key, i) => ({
        name: key,
        count: distribution[split][key],
        fill: COLORS[i % COLORS.length]
      }))
    : [];

  return (
    <div className="flex-1 flex flex-col overflow-y-auto bg-slate-950">
      <Header 
        title="AG News Dataset Explorer" 
        description="Inspect features, splits, and sample articles directly loaded from Hugging Face Mikami063/ag_news."
      />

      <main className="p-8 space-y-8">
        {/* Dataset Metadata Box */}
        <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center space-x-2">
              <h3 className="text-lg font-bold text-white">Repository: {info?.dataset_name || 'Mikami063/ag_news'}</h3>
              <a
                href={info?.dataset_url || 'https://huggingface.co/datasets/Mikami063/ag_news'}
                target="_blank"
                rel="noreferrer"
                className="text-cyan-400 hover:text-cyan-300 inline-flex items-center space-x-1 text-xs"
              >
                <span>Hugging Face</span>
                <ExternalLink className="w-3.5 h-3.5" />
              </a>
            </div>
            <p className="text-xs text-slate-400 mt-1">
              Standard benchmark for text classification containing news articles from AG's corpus.
            </p>
          </div>

          <div className="flex items-center space-x-4 text-xs">
            <div className="px-4 py-2 rounded-xl bg-slate-950 border border-slate-800">
              <span className="text-slate-500 block">Train Split</span>
              <span className="text-white font-semibold text-sm">{info?.train_records.toLocaleString() || '120,000'}</span>
            </div>
            <div className="px-4 py-2 rounded-xl bg-slate-950 border border-slate-800">
              <span className="text-slate-500 block">Official Test Split</span>
              <span className="text-white font-semibold text-sm">{info?.test_records.toLocaleString() || '7,600'}</span>
            </div>
          </div>
        </div>

        {/* Distribution Chart and Class Breakdown */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 p-6 rounded-2xl bg-slate-900/80 border border-slate-800">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h4 className="text-sm font-semibold text-white">Class Distribution ({split.toUpperCase()} Split)</h4>
                <p className="text-xs text-slate-400">Class balance across World, Sports, Business, and Sci/Tech</p>
              </div>
              <div className="flex bg-slate-950 p-1 rounded-lg border border-slate-800 text-xs">
                <button
                  onClick={() => { setSplit('train'); setPage(1); }}
                  className={"px-3 py-1 rounded-md font-medium transition-colors " + (split === 'train' ? 'bg-cyan-500/20 text-cyan-400' : 'text-slate-400 hover:text-white')}
                >
                  Train
                </button>
                <button
                  onClick={() => { setSplit('test'); setPage(1); }}
                  className={"px-3 py-1 rounded-md font-medium transition-colors " + (split === 'test' ? 'bg-cyan-500/20 text-cyan-400' : 'text-slate-400 hover:text-white')}
                >
                  Test
                </button>
              </div>
            </div>

            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData}>
                  <XAxis dataKey="name" stroke="#64748b" fontSize={11} />
                  <YAxis stroke="#64748b" fontSize={11} />
                  <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }} />
                  <Bar dataKey="count" radius={[6, 6, 0, 0]}>
                    {chartData.map((entry, index) => (
                      <Cell key={'cell-' + index} fill={entry.fill} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 flex flex-col justify-between">
            <div>
              <h4 className="text-sm font-semibold text-white mb-2">Category Mapping</h4>
              <p className="text-xs text-slate-400 mb-4">Verified label integer-to-string mapping</p>
              <div className="space-y-2">
                {[
                  { id: 0, name: 'World', desc: 'International affairs & diplomacy' },
                  { id: 1, name: 'Sports', desc: 'Athletic competitions & tournaments' },
                  { id: 2, name: 'Business', desc: 'Financial markets & commerce' },
                  { id: 3, name: 'Sci/Tech', desc: 'Science, hardware, software, AI' },
                ].map((item) => (
                  <div key={item.id} className="p-3 rounded-xl bg-slate-950/60 border border-slate-800 text-xs">
                    <div className="flex items-center justify-between">
                      <span className="font-semibold text-white">{item.name}</span>
                      <span className="px-2 py-0.5 rounded bg-slate-800 text-slate-300 font-mono text-[10px]">
                        ID: {item.id}
                      </span>
                    </div>
                    <span className="text-slate-500 text-[11px] mt-0.5 block">{item.desc}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Paginated Examples Table */}
        <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h4 className="text-sm font-semibold text-white">Sample News Records</h4>
              <p className="text-xs text-slate-400">Inspecting real raw samples from the {split} split</p>
            </div>
            <div className="flex items-center space-x-2 text-xs">
              <button
                onClick={() => setPage(p => Math.max(1, p - 1))}
                disabled={page === 1}
                className="p-1.5 rounded-lg bg-slate-800 text-slate-300 hover:text-white disabled:opacity-40"
              >
                <ChevronLeft className="w-4 h-4" />
              </button>
              <span className="text-slate-400">Page {page} of {Math.ceil((examples.total || 1) / 10)}</span>
              <button
                onClick={() => setPage(p => p + 1)}
                disabled={page >= Math.ceil((examples.total || 1) / 10)}
                className="p-1.5 rounded-lg bg-slate-800 text-slate-300 hover:text-white disabled:opacity-40"
              >
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          </div>

          <div className="space-y-3">
            {examples.items.map((ex) => (
              <div key={ex.id} className="p-4 rounded-xl bg-slate-950/70 border border-slate-800 text-xs flex flex-col md:flex-row md:items-start justify-between gap-3">
                <p className="text-slate-300 leading-relaxed flex-1">{ex.text}</p>
                <span className="self-start px-2.5 py-1 rounded-md bg-slate-800 border border-slate-700 text-cyan-400 font-semibold text-[11px] whitespace-nowrap">
                  {ex.category}
                </span>
              </div>
            ))}
          </div>
        </div>
      </main>
    </div>
  );
};

export default DatasetPage;
