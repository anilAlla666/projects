import React from 'react';
import { motion } from 'framer-motion';
import { ArrowDown } from 'lucide-react';

const TIRUSOMADeepDive = () => {
  return (
    <section className="py-24 bg-background">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-serif font-normal text-[#fafafa] mb-4">TIRUSOMA</h2>
          <p className="text-xl font-sans text-primary">The spinal cord for any robot</p>
        </div>

        <div className="grid md:grid-cols-2 gap-12 mb-24">
          <motion.div 
            initial={{ opacity: 0, x: -20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            className="bg-card p-8 rounded-xl border border-border shadow-lg"
          >
            <h3 className="text-2xl font-serif font-normal text-[#fafafa] mb-4">The Problem</h3>
            <p className="text-[#a1a1aa] font-sans leading-relaxed text-base">
              Modern robotics relies on hand-coded C++, Model Predictive Control (MPC), and complex kinematic equations that scale poorly. Computing balance, trajectory, and multi-joint actuation requires intense processing power, resulting in high latency, jittery movement, and the inability to adapt instantly to unforeseen physical perturbations. This traditional stack is brittle, expensive to maintain, and fundamentally limited by its iterative nature.
            </p>
          </motion.div>

          <motion.div 
            initial={{ opacity: 0, x: 20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            className="bg-card p-8 rounded-xl border border-primary/30 relative overflow-hidden shadow-lg"
          >
            <div className="absolute inset-0 bg-primary/5 pointer-events-none" />
            <h3 className="text-2xl font-serif font-normal text-[#fafafa] mb-4 relative z-10">The Solution</h3>
            <p className="text-[#a1a1aa] font-sans leading-relaxed text-base relative z-10">
              TIRUSOMA replaces thousands of lines of C++ with a single, highly optimized CfC network. It calculates exact inverse kinematics and dynamics in O(1) time. By mapping the entire physical possibility space into a queryable latent structure, it provides instant, fluid, and biologically-accurate motor commands. This unified approach collapses the entire stack into one robust, mathematically guaranteed forward pass.
            </p>
          </motion.div>
        </div>

        <div className="max-w-4xl mx-auto">
          <h4 className="text-center font-sans text-muted-foreground uppercase tracking-widest text-sm font-semibold mb-12">Unified Architecture</h4>
          
          <div className="flex flex-col items-center">
            <motion.div 
              initial={{ opacity: 0, y: -20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              className="w-full max-w-md bg-card border border-border rounded-xl p-6 text-center shadow-lg relative z-10"
            >
              <div className="font-sans font-semibold text-[#fafafa] text-lg">Any High-Level Planner</div>
              <div className="font-sans text-sm text-[#a1a1aa] mt-2">(VLM / LLM / User Input / Teleoperation)</div>
            </motion.div>

            <FlowArrow label="target command" />

            <motion.div 
              initial={{ opacity: 0, scale: 0.95 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              transition={{ delay: 0.2, duration: 0.5 }}
              className="w-full max-w-2xl bg-card border-2 border-primary rounded-xl p-10 text-center shadow-[0_0_40px_rgba(245,158,11,0.15)] relative z-10"
            >
              <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,_rgba(245,158,11,0.1)_0%,_transparent_70%)] pointer-events-none rounded-xl" />
              
              <h3 className="text-4xl md:text-5xl font-serif font-normal text-[#fafafa] mb-2 relative z-10 tracking-wide">TIRUSOMA</h3>
              <p className="text-xl md:text-2xl font-sans font-medium text-primary mb-6 relative z-10">Universal Motor Runtime</p>
              
              <div className="inline-block bg-background border border-border rounded-lg px-4 py-2 relative z-10">
                <p className="font-mono text-sm md:text-base text-[#fafafa]">
                  244KB <span className="text-[#a1a1aa] mx-1">·</span> 
                  10kHz on CPU <span className="text-[#a1a1aa] mx-1">·</span> 
                  Constant time <span className="text-[#a1a1aa] mx-1">·</span> 
                  Safety-certified
                </p>
              </div>
            </motion.div>

            <FlowArrow label="safe torques" />

            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.4 }}
              className="w-full max-w-md bg-card border border-border rounded-xl p-6 text-center shadow-lg relative z-10"
            >
              <div className="font-sans font-semibold text-[#fafafa] text-lg">Motors</div>
              <div className="font-sans text-sm text-[#a1a1aa] mt-2">Any robot. Any morphology.</div>
            </motion.div>

          </div>
        </div>
      </div>
    </section>
  );
};

const FlowArrow = ({ label }) => (
  <div className="flex flex-col items-center justify-start h-20 w-full relative -my-1 z-0">
    <motion.div 
      initial={{ height: 0 }}
      whileInView={{ height: "100%" }}
      viewport={{ once: true }}
      transition={{ duration: 0.6, ease: "easeOut" }}
      className="w-[2px] bg-primary/80 relative"
    >
      <motion.div 
        initial={{ opacity: 0, x: -10 }}
        whileInView={{ opacity: 1, x: 0 }}
        viewport={{ once: true }}
        transition={{ delay: 0.4 }}
        className="absolute left-6 top-1/2 -translate-y-1/2"
      >
        <div className="bg-background border border-primary/30 px-3 py-1.5 rounded text-xs font-mono text-primary uppercase tracking-widest whitespace-nowrap shadow-md">
          {label}
        </div>
      </motion.div>
    </motion.div>
    
    <motion.div 
      initial={{ opacity: 0 }}
      whileInView={{ opacity: 1 }}
      viewport={{ once: true }}
      transition={{ delay: 0.5 }}
      className="text-primary absolute bottom-[-10px]"
    >
      <ArrowDown className="w-5 h-5" />
    </motion.div>
  </div>
);

export default TIRUSOMADeepDive;