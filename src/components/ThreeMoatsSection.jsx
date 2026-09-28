import React from 'react';
import { motion } from 'framer-motion';

const ThreeMoatsSection = () => {
  const moats = [
    {
      title: "Constant-Time Inference",
      desc: "Our O(1) architecture ensures that scaling complexity (e.g., adding more joints or constraints) does not increase compute time. Competitors relying on iterative solvers face exponentially rising latency."
    },
    {
      title: "Frequency Invariance",
      desc: "TIRUSOMA’s continuous-time nature allows it to query exact physical states at any frequency without retraining, providing infinite resolution for control systems."
    },
    {
      title: "Certifiable Safety",
      desc: "Unlike black-box LLMs or RL policies, our system incorporates native Control Barrier Functions (CBFs) ensuring mathematically guaranteed safety constraints at 84kHz."
    }
  ];

  return (
    <section className="py-24 bg-[#09090b]">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <h2 className="text-3xl md:text-4xl font-serif font-normal text-white mb-16 text-center">Why This Is Hard to Copy</h2>
        
        <div className="grid md:grid-cols-3 gap-8">
          {moats.map((moat, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, scale: 0.95 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              className="bg-[#18181b] p-8 rounded-2xl border border-[#27272a] relative overflow-hidden"
            >
              <div className="absolute top-0 left-0 w-full h-1 bg-[#f59e0b]" />
              <h3 className="text-xl font-serif font-normal text-white mb-4 mt-2">{moat.title}</h3>
              <p className="text-gray-400 font-sans leading-relaxed text-sm">{moat.desc}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default ThreeMoatsSection;