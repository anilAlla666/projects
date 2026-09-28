import React from 'react';
import { motion } from 'framer-motion';
import ScrollAnimatedSection from '@/components/ScrollAnimatedSection';

const CapabilitiesGrid = () => {
  const capabilities = [
    { title: "Real-time Control", desc: "Instantaneous execution of motor commands with zero perceptible latency, ensuring fluid and natural robotic movement in dynamic environments." },
    { title: "Adaptive Learning", desc: "Dynamically adjusts to unknown surface friction, elevations, and external perturbations without requiring complete model retraining." },
    { title: "Energy Efficiency", desc: "Optimized O(1) inference allows for complex behavior generation on low-power microcontrollers, significantly extending operational uptime." },
    { title: "Fault Tolerance", desc: "Calculated multi-contact recovery and safe fallback states prevent catastrophic failures during unstable physical interactions." },
    { title: "Scalability", desc: "A unified architecture that scales seamlessly across different robot morphologies and use cases without changing the core runtime." },
    { title: "Integration", desc: "Provides simple, safe torque outputs that can be easily consumed by any high-level planner, LLM, or user teleoperation system." },
  ];

  return (
    <section className="py-24 bg-background">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <ScrollAnimatedSection>
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-serif font-normal text-[#fafafa] mb-4">21 Core Motor Behaviors</h2>
            <p className="text-lg font-sans text-[#a1a1aa]">One network. Trained in <span className="text-[#f59e0b] font-mono">91 minutes</span>. Deployed at <span className="text-[#f59e0b] font-mono">244KB</span>.</p>
          </div>
        </ScrollAnimatedSection>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {capabilities.map((cap, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.1, duration: 0.5 }}
              whileHover={{ scale: 1.02 }}
              className="bg-card border border-border hover:border-l-[3px] hover:border-l-[#f59e0b] p-8 rounded-xl transition-all duration-300 shadow-lg group"
            >
              <h3 className="text-xl font-serif font-normal text-[#fafafa] mb-3 group-hover:text-[#f59e0b] transition-colors">{cap.title}</h3>
              <p className="text-[#a1a1aa] font-sans text-base leading-relaxed">{cap.desc}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default CapabilitiesGrid;