import { useScroll, useTransform } from 'framer-motion';
import { useRef } from 'react';

export default function useParallax(offset = 0.3) {
  const ref = useRef(null);
  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  
  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ["start end", "end start"]
  });

  const y = useTransform(
    scrollYProgress,
    [0, 1],
    prefersReducedMotion ? ["0%", "0%"] : [`-${offset * 100}%`, `${offset * 100}%`]
  );

  return { ref, y };
}