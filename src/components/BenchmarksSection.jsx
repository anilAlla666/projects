import React from 'react';
import useCountUp from '@/hooks/useCountUp';
import ScrollAnimatedSection from '@/components/ScrollAnimatedSection';

const StatCard = ({ endValue, suffix, label }) => {
  const { count, ref } = useCountUp(endValue, 1.5);

  return (
    <div ref={ref} className="bg-card border border-border p-8 rounded-xl shadow-lg text-center hover:shadow-[0_10px_30px_rgba(245,158,11,0.1)] transition-shadow duration-300">
      <div className="text-4xl md:text-5xl font-mono font-bold text-[#f59e0b] mb-2">
        {count}{suffix}
      </div>
      <div className="text-sm font-sans font-medium text-[#a1a1aa] uppercase tracking-wider">{label}</div>
    </div>
  );
};

const BenchmarksSection = ({ id }) => {
  return (
    <section id={id} className="py-24 relative bg-card/30 border-y border-border">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <ScrollAnimatedSection>
          <h2 className="text-3xl md:text-4xl font-serif font-normal text-[#fafafa] mb-12 text-center">Engineered for Performance</h2>
        </ScrollAnimatedSection>
        
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
          <ScrollAnimatedSection staggerDelay={0.1}>
            <StatCard endValue={244} suffix="KB" label="Deployment Size" />
          </ScrollAnimatedSection>
          <ScrollAnimatedSection staggerDelay={0.2}>
            <StatCard endValue={10} suffix="kHz" label="Control Frequency" />
          </ScrollAnimatedSection>
          <ScrollAnimatedSection staggerDelay={0.3}>
            <StatCard endValue={21} suffix="" label="Core Behaviors" />
          </ScrollAnimatedSection>
          <ScrollAnimatedSection staggerDelay={0.4}>
            <StatCard endValue={91} suffix=" min" label="Training Time" />
          </ScrollAnimatedSection>
        </div>
      </div>
    </section>
  );
};

export default BenchmarksSection;