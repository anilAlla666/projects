import React from 'react';
import { motion } from 'framer-motion';
import { Gamepad2, Cpu } from 'lucide-react';

const CompanyThesisSection = ({ id }) => {
  return (
    <section id={id} className="py-24 bg-background relative overflow-hidden">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-serif font-normal text-[#fafafa] mb-4">One Thesis. Two Frontiers.</h2>
          <p className="text-xl text-[#a1a1aa] max-w-3xl font-sans mx-auto">
            Neural Dynamics Team replaces iterative solvers with constant-time neural inference — across gaming, robotics, and beyond.
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-8">
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="group bg-card border border-border hover:border-primary transition-colors duration-300 rounded-xl p-8 shadow-lg"
          >
            <div className="flex items-center gap-3 mb-4">
              <Gamepad2 className="text-primary w-8 h-8" />
              <h3 className="text-3xl font-serif font-normal text-[#fafafa]">HyperFlux</h3>
            </div>
            <h4 className="text-lg font-sans font-medium text-[#fafafa] mb-6">Efficient Transformer Scaling</h4>
            <p className="text-[#a1a1aa] font-sans mb-8 leading-relaxed text-base">
              We are bringing O(1) computational kernel to AAA gaming. Simulate millions of rigid bodies, fluid dynamics, and complex destructible environments simultaneously with zero frame-drops. Decouple physics from render loops entirely.
            </p>
            <div className="grid grid-cols-2 gap-6">
              {[
                { label: "Performance Gain", value: "2.7x Faster" },
                { label: "Memory Usage", value: "60% Less Memory" },
                { label: "Processing", value: "3x Throughput" },
                { label: "Response", value: "50% Latency Reduction" },
              ].map((stat, i) => (
                <div key={i}>
                  <div className="text-lg font-mono font-bold text-[#fafafa]">{stat.value}</div>
                  <div className="text-sm font-sans text-[#a1a1aa]">{stat.label}</div>
                </div>
              ))}
            </div>
          </motion.div>

          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.1 }}
            className="group bg-card border border-border hover:border-primary transition-colors duration-300 rounded-xl p-8 shadow-lg"
          >
            <div className="flex items-center gap-3 mb-4">
              <Cpu className="text-primary w-8 h-8" />
              <h3 className="text-3xl font-serif font-normal text-[#fafafa]">TIRUSOMA</h3>
            </div>
            <h4 className="text-lg font-sans font-medium text-[#fafafa] mb-6">Universal Motor Runtime</h4>
            <p className="text-[#a1a1aa] font-sans mb-8 leading-relaxed text-base">
              A neural motor cortex that utilizes phase-space encoding to calculate exact inverse kinematics and dynamics in O(1) time. By mapping the entire physical possibility space into a queryable latent structure, TIRUSOMA provides instant, fluid motor commands.
            </p>
            <div className="grid grid-cols-2 gap-6">
              {[
                { label: "Deployment Size", value: "244KB" },
                { label: "Control Freq", value: "10kHz" },
                { label: "Reliability", value: "99.9% Uptime" },
                { label: "Performance", value: "Sub-ms Latency" },
              ].map((stat, i) => (
                <div key={i}>
                  <div className="text-lg font-mono font-bold text-[#fafafa]">{stat.value}</div>
                  <div className="text-sm font-sans text-[#a1a1aa]">{stat.label}</div>
                </div>
              ))}
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
};

export default CompanyThesisSection;