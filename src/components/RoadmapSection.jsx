import React from 'react';
import { motion } from 'framer-motion';
import ScrollAnimatedSection from '@/components/ScrollAnimatedSection';

const RoadmapSection = () => {
  const items = [
    { version: "v14", status: "completed", title: "Core CfC Architecture Validation", desc: "Simulated bipedal walking at 0.02ms latency." },
    { version: "v14.1", status: "progress", title: "TIRUSOMA Alpha Integration", desc: "Real-world testing on quadrupedal hardware." },
    { version: "v15", status: "upcoming", title: "Full Omnidirectional Suite", desc: "21 behaviors certified across multiple morphologies." },
    { version: "v16", status: "upcoming", title: "HyperFlux Gaming Beta", desc: "SDK release for AAA engine physics integration." },
    { version: "Runtime SDK", status: "upcoming", title: "Public Enterprise Release", desc: "Self-serve deployment for edge robotics." },
  ];

  return (
    <section className="py-24 bg-background">
      <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8">
        <ScrollAnimatedSection>
          <h2 className="text-3xl font-serif font-normal text-[#fafafa] mb-16 text-center">Development Roadmap</h2>
        </ScrollAnimatedSection>
        
        <div className="space-y-8 relative">
          <motion.div 
            initial={{ height: 0 }}
            whileInView={{ height: "100%" }}
            viewport={{ once: true }}
            transition={{ duration: 1.5, ease: "linear" }}
            className="absolute left-5 md:left-1/2 md:-ml-[1px] top-0 w-0.5 bg-gradient-to-b from-transparent via-border to-transparent z-0"
          />
          
          {items.map((item, i) => (
            <motion.div 
              key={i} 
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ delay: i * 0.15 }}
              className="relative flex items-center justify-between md:justify-normal md:odd:flex-row-reverse group z-10"
            >
              
              <div className={`flex items-center justify-center w-10 h-10 rounded-full border-4 border-background shadow shrink-0 md:order-1 md:group-odd:-translate-x-1/2 md:group-even:translate-x-1/2 transition-shadow duration-300 hover:shadow-[0_0_15px_rgba(245,158,11,0.8)] ${
                item.status === 'completed' ? 'bg-[#f59e0b]' : 
                item.status === 'progress' ? 'bg-[#f59e0b] animate-pulse shadow-[0_0_15px_rgba(245,158,11,0.6)]' : 
                'bg-card border-border'
              }`}>
              </div>
              
              <div className="w-[calc(100%-4rem)] md:w-[calc(50%-2.5rem)] p-6 rounded-xl bg-card border border-border hover:border-[#f59e0b]/50 transition-colors">
                <div className="flex items-center justify-between mb-1">
                  <h3 className="font-serif font-normal text-[#fafafa] text-lg">{item.title}</h3>
                  <span className={`text-xs font-mono font-medium px-2 py-1 rounded ${
                    item.status === 'completed' || item.status === 'progress' ? 'text-[#f59e0b] bg-[#f59e0b]/10' : 'text-[#a1a1aa] bg-white/5'
                  }`}>{item.version}</span>
                </div>
                <p className="text-sm font-sans text-[#a1a1aa]">{item.desc}</p>
              </div>
              
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default RoadmapSection;