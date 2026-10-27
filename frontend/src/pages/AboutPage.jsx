import React from 'react';
import Header from '../components/Header';
import { BookOpen, Cpu, Database, Network, GitBranch, ShieldAlert } from 'lucide-react';

const AboutPage = () => {
  return (
    <div className="flex-1 flex flex-col overflow-y-auto bg-slate-950">
      <Header 
        title="About the Machine Learning Project" 
        description="ANN-Based Classification of AG News Using Continuous Bag of Words (CBOW) Embedding + Multi-Layer Perceptron (MLP)."
      />

      <main className="p-8 max-w-5xl space-y-8">
        {/* Project Statement */}
        <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-3">
          <div className="flex items-center space-x-2 text-cyan-400">
            <BookOpen className="w-5 h-5" />
            <h3 className="text-base font-bold text-white">Problem Statement & Objectives</h3>
          </div>
          <p className="text-xs text-slate-300 leading-relaxed">
            The AG News corpus is a landmark dataset in Natural Language Processing consisting of news articles categorized into four key domains: <strong>World, Sports, Business, and Sci/Tech</strong>. The objective of this project is to construct a complete, end-to-end Neural Network architecture that learns distributed semantic representations through a custom <strong>Continuous Bag of Words (CBOW)</strong> model trained on sliding context windows, and uses these learned embeddings within a <strong>Multi-Layer Perceptron (MLP)</strong> classifier trained with <strong>Mini-Batch Gradient Descent</strong>.
          </p>
        </div>

        {/* Complete Architecture Diagram */}
        <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-4">
          <div className="flex items-center space-x-2 text-indigo-400">
            <Network className="w-5 h-5" />
            <h3 className="text-base font-bold text-white">Model Architecture & Data Flow</h3>
          </div>

          <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 font-mono text-xs text-slate-300 space-y-2 overflow-x-auto">
            <p className="text-cyan-400 font-bold">[1] Input Raw News Text</p>
            <p className="pl-4">↓ Preprocessing (Regex tokenization, lowercasing, HTML unescape, &lt;PAD&gt;=0, &lt;UNK&gt;=1)</p>
            <p className="text-indigo-400 font-bold">[2] Integer Token Sequence [B, Seq_Len]</p>
            <p className="pl-4">↓ CBOW Pre-trained Embedding Lookup Matrix [V × D]</p>
            <p className="text-purple-400 font-bold">[3] Token Vectors [B, Seq_Len, D]</p>
            <p className="pl-4">↓ Padding-Aware Masked Mean Pooling (zeroing &lt;PAD&gt; tokens)</p>
            <p className="text-emerald-400 font-bold">[4] Sentence Representation Vector [B, D]</p>
            <p className="pl-4">↓ Dense Linear Layer 1 + ReLU Activation + Dropout (p=0.3)</p>
            <p className="text-amber-400 font-bold">[5] Hidden Representation [B, H]</p>
            <p className="pl-4">↓ Linear Output Projection [B, 4]</p>
            <p className="text-emerald-300 font-bold">[6] Logits → Softmax Probabilities: [World, Sports, Business, Sci/Tech]</p>
          </div>
        </div>

        {/* Methodological Details Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-3">
            <div className="flex items-center space-x-2 text-cyan-400">
              <Cpu className="w-5 h-5" />
              <h4 className="text-sm font-bold text-white">CBOW Embedding Pretraining</h4>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              Unlike static TF-IDF vectors, CBOW models learn continuous vector representations by using surrounding context words (window size <em>w</em>) to predict the target center word. The learned weight matrix acts as a rich semantic space where words with similar contextual distributions map to nearby vectors.
            </p>
          </div>

          <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-3">
            <div className="flex items-center space-x-2 text-indigo-400">
              <GitBranch className="w-5 h-5" />
              <h4 className="text-sm font-bold text-white">Mini-Batch Gradient Descent (SGD)</h4>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              The academic specification emphasizes Mini-Batch Gradient Descent. Weights are iteratively updated using mini-batches (e.g. 64) with momentum (0.9), balancing computational efficiency on CPU/GPU with smooth stochastic convergence compared to single-example SGD or full batch gradient descent.
            </p>
          </div>

          <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-3">
            <div className="flex items-center space-x-2 text-emerald-400">
              <Database className="w-5 h-5" />
              <h4 className="text-sm font-bold text-white">Dataset Protocol & Integrity</h4>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              Strict scientific evaluation protocols are enforced: the 7,600 test examples from Hugging Face <code>Mikami063/ag_news</code> are strictly preserved for final evaluation. Only the training partition is used for vocabulary generation, context pair sampling, and validation splitting (85/15), preventing any data snooping or leakage.
            </p>
          </div>

          <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-3">
            <div className="flex items-center space-x-2 text-amber-400">
              <ShieldAlert className="w-5 h-5" />
              <h4 className="text-sm font-bold text-white">Baseline vs Proposed Trade-offs</h4>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              The TF-IDF baseline represents articles using sparse word frequency statistics without word order or semantic vector similarity. The proposed CBOW + MLP pipeline captures word proximity and dense embeddings while remaining significantly faster to train and evaluate than modern heavy transformer models.
            </p>
          </div>
        </div>
      </main>
    </div>
  );
};

export default AboutPage;
