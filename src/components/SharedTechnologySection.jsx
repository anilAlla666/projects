import React from 'react';
import { motion } from 'framer-motion';
import { Network, Zap, Cpu } from 'lucide-react';

const SharedTechnologySection = () => {
  const items = [
    {
      icon: Network,
      title: "Closed-form Continuous-time Networks (CfC)",
      description: "Unlike traditional RNNs or Transformers, our architecture models continuous physical dynamics without discrete time stepping, ensuring perfect phase-space alignment and infinite resolution."
    },
    {
      icon: Zap,
      title: "O(1) Constant-Time Inference",
      description: "By replacing iterative solvers with a single forward pass, computation time becomes strictly bounded. This ultra-low latency approach guarantees predictable performance regardless of scene complexity."
    },
    {
      icon: Cpu,
      title: "INT8 Deployment",
      description: "Aggressively quantized for edge hardware. Runs completely offline on standard microcontrollers without requiring heavy cloud GPU compute, bringing advanced intelligence directly to the device."
    }
  ];

  return (
    <section className="py-24 bg-background relative">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-serif font-normal text-[#fafafa]">The Shared Core</h2>
        </div>
        
        <div className="grid md:grid-cols-3 gap-8">
          {items.map((item, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.1 }}
              className="bg-card border border-border rounded-xl p-8 hover:border-primary transition-colors shadow-lg"
            >
              <item.icon className="w-10 h-10 text-primary mb-6" />
              <h3 className="text-xl font-serif font-normal text-[#fafafa] mb-4 leading-tight">{item.title}</h3>
              <p className="text-[#a1a1aa] font-sans leading-relaxed text-base">
                {item.description}
              </p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default SharedTechnologySection;