import React from 'react';
import { motion } from 'framer-motion';

const MissionStatement = ({ id }) => {
  return (
    <section className="py-32 relative overflow-hidden flex items-center justify-center min-h-[60vh]" id={id}>
      {/* Animated background elements */}
      <div className="absolute inset-0 bg-navy-950">
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-gold-500/5 rounded-full blur-[120px] animate-pulse" />
        <div className="absolute top-0 left-0 w-full h-full bg-[radial-gradient(circle_at_50%_50%,_rgba(15,23,42,0)_0%,_rgba(15,23,42,1)_100%)] z-10" />
      </div>

      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 relative z-20 text-center">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 1 }}
        >
          <h2 className="text-sm font-bold text-gold-500 uppercase tracking-[0.3em] mb-8">
            Our Vision
          </h2>
          
          <h1 className="text-4xl md:text-6xl lg:text-7xl font-bold leading-tight mb-8">
            To solve the <span className="text-transparent bg-clip-text bg-gradient-to-r from-white via-gold-200 to-white animate-shimmer bg-[length:200%_auto]">latency gap</span> between human thought and digital reality.
          </h1>
          
          <p className="text-xl md:text-2xl text-gray-400 max-w-3xl mx-auto font-light leading-relaxed">
            We believe that instant computation is not just a performance metric—it is the catalyst for the next era of intelligence, where software operates at the speed of physics.
          </p>
        </motion.div>
      </div>
    </section>
  );
};

export default MissionStatement;