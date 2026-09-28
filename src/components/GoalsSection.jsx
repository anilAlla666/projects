import React, { useRef } from 'react';
import { motion, useScroll, useTransform } from 'framer-motion';
import { Rocket, Cloud, Server } from 'lucide-react';
import TimelineItem from './TimelineItem';

const GoalsSection = ({ id }) => {
  const containerRef = useRef(null);
  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start end", "end start"]
  });

  const opacity = useTransform(scrollYProgress, [0, 0.2], [0, 1]);
  const y = useTransform(scrollYProgress, [0, 0.2], [100, 0]);

  const phases = [
    {
      phase: 1,
      title: "Real-time Simulation",
      description: "Implementing O(1) kernels for AAA gaming engines and robotics control systems where latency < 1ms is critical.",
      icon: Rocket
    },
    {
      phase: 2,
      title: "Cloud Expansion",
      description: "Deploying distributed HyperFlux instances across major cloud providers to enable infinite-scale physics simulations.",
      icon: Cloud
    },
    {
      phase: 3,
      title: "Hardware Evolution",
      description: "Direct silicon implementation of phase-space encoding for specialized GPU architectures and edge computing units.",
      icon: Server
    }
  ];

  return (
    <section ref={containerRef} className="py-32 relative overflow-hidden bg-[#09090b]" id={id}>
      <div className="absolute inset-0 bg-gradient-to-b from-[#09090b] via-[#18181b] to-[#09090b] opacity-50 pointer-events-none" />
      
      <motion.div 
        className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10"
        style={{ opacity, y }}
      >
        <div className="text-center mb-20">
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-6">
            Strategic Roadmap
          </h2>
          <p className="text-xl text-gray-400 max-w-2xl mx-auto">
            Our path to redefining the limits of computational physics.
          </p>
        </div>

        <div className="relative">
          {/* Central Timeline Line (Desktop) */}
          <div className="hidden md:block absolute left-1/2 top-0 bottom-0 w-px bg-gradient-to-b from-transparent via-[#f59e0b]/50 to-transparent -translate-x-1/2" />

          <div className="space-y-12 md:space-y-24">
            {phases.map((phase, index) => (
              <div key={index} className={`flex flex-col md:flex-row gap-8 items-center ${index % 2 === 0 ? 'md:flex-row-reverse' : ''}`}>
                <div className="flex-1 w-full">
                  <TimelineItem {...phase} />
                </div>
                
                {/* Timeline Node */}
                <div className="relative z-10 hidden md:flex items-center justify-center w-12 h-12 rounded-full bg-[#09090b] border-2 border-[#f59e0b] shadow-[0_0_20px_rgba(245,158,11,0.3)]">
                  <div className="w-3 h-3 rounded-full bg-[#fbbf24] animate-pulse" />
                </div>
                
                <div className="flex-1 w-full hidden md:block" />
              </div>
            ))}
          </div>
        </div>
      </motion.div>
    </section>
  );
};

export default GoalsSection;