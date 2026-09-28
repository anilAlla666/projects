import React from 'react';
import { motion } from 'framer-motion';
import { Bot, Factory, ActivitySquare } from 'lucide-react';
import ScrollAnimatedSection from '@/components/ScrollAnimatedSection';

const applications = [
  { icon: Factory, title: "Industrial Automation", desc: "Reliable, deterministic control for manufacturing lines requiring sub-millisecond precision." },
  { icon: Bot, title: "Autonomous Systems", desc: "Adaptive stabilization for bipeds and quadrupeds across unpredictable terrain." },
  { icon: ActivitySquare, title: "Precision Manufacturing", desc: "Micro-adjustments and dynamic force compensation natively executed at the edge." }
];

const RoboticsApplicationsSection = ({ id }) => {
  return (
    <section id={id} className="py-24 relative bg-card/50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <ScrollAnimatedSection>
          <h2 className="text-3xl md:text-4xl font-serif font-normal text-[#fafafa] mb-12 text-center">Robotics Applications</h2>
        </ScrollAnimatedSection>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {applications.map((app, i) => (
            <ScrollAnimatedSection key={i} staggerDelay={0.1 * i}>
              <motion.div
                whileHover={{ scale: 1.02 }}
                className="bg-background border border-border p-8 rounded-xl shadow-lg hover:border-l-4 hover:border-l-[#f59e0b] transition-all duration-200 h-full"
              >
                <app.icon className="text-[#f59e0b] w-8 h-8 mb-4" />
                <h3 className="text-xl font-serif font-normal text-[#fafafa] mb-3">{app.title}</h3>
                <p className="text-[#a1a1aa] font-sans text-base leading-relaxed">{app.desc}</p>
              </motion.div>
            </ScrollAnimatedSection>
          ))}
        </div>
      </div>
    </section>
  );
};

export default RoboticsApplicationsSection;