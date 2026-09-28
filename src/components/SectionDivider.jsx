import React, { useRef } from 'react';
import { motion } from 'framer-motion';
import useScrollAnimation from '@/hooks/useScrollAnimation';

const SectionDivider = () => {
  const ref = useRef(null);
  const isVisible = useScrollAnimation(ref);
  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  return (
    <div ref={ref} className="w-full flex justify-center py-12">
      <motion.div
        className="h-[1px] bg-gradient-to-r from-transparent via-[#f59e0b]/50 to-transparent"
        initial={prefersReducedMotion ? { width: "100%" } : { width: "0%" }}
        animate={isVisible ? { width: "100%" } : {}}
        transition={{ duration: 1, ease: "easeOut" }}
      />
    </div>
  );
};

export default SectionDivider;