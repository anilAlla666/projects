import React, { useMemo } from 'react';

const FloatingParticles = () => {
  const particles = useMemo(() => {
    return Array.from({ length: 20 }).map((_, i) => ({
      id: i,
      left: `${Math.random() * 100}vw`,
      animationDuration: `${2 + Math.random() * 3}s`,
      animationDelay: `${Math.random() * 2}s`,
      size: `${2 + Math.random()}px`
    }));
  }, []);

  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return null;

  return (
    <div className="fixed inset-0 z-[-10] pointer-events-none overflow-hidden">
      {particles.map((p) => (
        <div
          key={p.id}
          className="absolute bg-[#f59e0b] rounded-full animate-float-up opacity-0"
          style={{
            left: p.left,
            width: p.size,
            height: p.size,
            animationDuration: p.animationDuration,
            animationDelay: p.animationDelay
          }}
        />
      ))}
    </div>
  );
};

export default FloatingParticles;