import React from 'react';
import { motion } from 'framer-motion';
const TeamSection = () => {
  return <section className="py-24 bg-background">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
        <h2 className="text-3xl md:text-4xl font-serif font-normal text-[#fafafa] mb-12">Team</h2>
        
        <motion.div initial={{
        opacity: 0,
        y: 20
      }} whileInView={{
        opacity: 1,
        y: 0
      }} viewport={{
        once: true
      }} className="bg-card border border-border rounded-xl p-10 max-w-2xl mx-auto shadow-lg">
          <h3 className="text-2xl font-serif font-normal text-[#fafafa] mb-2">Anil K Alla — Founder & CTO</h3>
          <p className="text-primary font-sans font-medium mb-6">Neural Dynamics Team</p>
          
          <p className="text-[#a1a1aa] font-sans leading-relaxed text-base">
            Previously: Research Scientist at Qwen/Alibaba (transformer architectures, MoE efficiency), Founding Researcher at Zhipu AI (co-developed GLM-130B, built CUDA kernels), DeepMind. Have co built first ever LLMs in China before GPT-3.
          </p>
        </motion.div>
      </div>
    </section>;
};
export default TeamSection;