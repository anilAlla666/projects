import React, { useRef } from 'react';
import { motion } from 'framer-motion';
import useScrollAnimation from '@/hooks/useScrollAnimation';

const TextRevealHeading = ({ children, className = '' }) => {
  const ref = useRef(null);
  const isVisible = useScrollAnimation(ref);
  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  return (
    <div ref={ref} className={`relative inline-block overflow-hidden ${className}`}>
      <motion.div
        initial={prefersReducedMotion ? { x: "100%" } : { x: "0%" }}
        animate={isVisible ? { x: "100%" } : {}}
        transition={{ duration: 0.6, ease: "circInOut" }}
        className="absolute inset-0 z-10 bg-[#f59e0b]"
      />
      <motion.div
        initial={prefersReducedMotion ? { opacity: 1 } : { opacity: 0 }}
        animate={isVisible ? { opacity: 1 } : {}}
        transition={{ duration: 0.1, delay: 0.3 }}
      >
        {children}
      </motion.div>
    </div>
  );
};

export default TextRevealHeading;