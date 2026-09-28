import React from 'react';
import { motion } from 'framer-motion';
import { Zap, Server, ShieldCheck } from 'lucide-react';
import ScrollAnimatedSection from '@/components/ScrollAnimatedSection';

const capabilities = [
  { icon: Zap, title: "Real-time Processing", desc: "Decouple massive physics simulations from rendering to guarantee steady frame rates." },
  { icon: Server, title: "Low Latency Architecture", desc: "O(1) execution bound ensures strict predictability for complex multiplayer calculations." },
  { icon: ShieldCheck, title: "High Performance Engine", desc: "Handles destructive environments, fluid dynamics, and cloth simulation natively." }
];

const GamingCapabilitiesSection = ({ id }) => {
  return (
    <section id={id} className="py-24 relative">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <ScrollAnimatedSection>
          <h2 className="text-3xl md:text-4xl font-serif font-normal text-[#fafafa] mb-12 text-center">Gaming Capabilities</h2>
        </ScrollAnimatedSection>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {capabilities.map((cap, i) => (
            <ScrollAnimatedSection key={i} staggerDelay={0.1 * i}>
              <motion.div
                whileHover={{ scale: 1.02, borderColor: "rgba(245, 158, 11, 0.5)" }}
                className="bg-card border border-border p-8 rounded-xl shadow-lg transition-all duration-300 flex flex-col items-center text-center"
              >
                <div className="p-4 bg-[#f59e0b]/10 rounded-full mb-6">
                  <cap.icon className="text-[#f59e0b] w-8 h-8" />
                </div>
                <h3 className="text-xl font-serif font-normal text-[#fafafa] mb-3">{cap.title}</h3>
                <p className="text-[#a1a1aa] font-sans text-base leading-relaxed">{cap.desc}</p>
              </motion.div>
            </ScrollAnimatedSection>
          ))}
        </div>
      </div>
    </section>
  );
};

export default GamingCapabilitiesSection;