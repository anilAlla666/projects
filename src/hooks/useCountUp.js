import { useState, useEffect, useRef } from 'react';
import useScrollAnimation from './useScrollAnimation';

export default function useCountUp(end, duration = 1.5) {
  const [count, setCount] = useState(0);
  const ref = useRef(null);
  const isVisible = useScrollAnimation(ref);
  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  useEffect(() => {
    if (!isVisible) return;
    if (prefersReducedMotion) {
      setCount(end);
      return;
    }

    let startTimestamp = null;
    const step = (timestamp) => {
      if (!startTimestamp) startTimestamp = timestamp;
      const progress = Math.min((timestamp - startTimestamp) / (duration * 1000), 1);
      
      // Ease out cubic
      const easeOut = 1 - Math.pow(1 - progress, 3);
      setCount(Math.floor(easeOut * end));

      if (progress < 1) {
        window.requestAnimationFrame(step);
      } else {
        setCount(end);
      }
    };
    
    window.requestAnimationFrame(step);
  }, [end, duration, isVisible, prefersReducedMotion]);

  return { count, ref };
}