import React from 'react';
import { motion, useScroll, useTransform } from 'framer-motion';

const GeometricShape = ({ className, style, type = 'cube' }) => {
  return (
    <motion.div className={`absolute preserve-3d ${className}`} style={style}>
      {type === 'cube' && (
        <div className="relative w-full h-full preserve-3d">
          <div className="absolute inset-0 bg-gold-500/10 border border-gold-500/30 backface-hidden translate-z-[50px]" />
          <div className="absolute inset-0 bg-gold-500/10 border border-gold-500/30 backface-hidden rotate-y-180 translate-z-[50px]" />
          <div className="absolute inset-0 bg-gold-500/10 border border-gold-500/30 backface-hidden rotate-x-90 translate-z-[50px]" />
          <div className="absolute inset-0 bg-gold-500/10 border border-gold-500/30 backface-hidden rotate-x--90 translate-z-[50px]" />
          <div className="absolute inset-0 bg-gold-500/10 border border-gold-500/30 backface-hidden rotate-y-90 translate-z-[50px]" />
          <div className="absolute inset-0 bg-gold-500/10 border border-gold-500/30 backface-hidden rotate-y--90 translate-z-[50px]" />
        </div>
      )}
      {type === 'orb' && (
        <div className="w-full h-full rounded-full bg-gradient-to-br from-gold-500/20 to-navy-900/20 blur-xl" />
      )}
    </motion.div>
  );
};

const ThreeDBackground = () => {
  const { scrollY } = useScroll();
  const y1 = useTransform(scrollY, [0, 2000], [0, 500]);
  const y2 = useTransform(scrollY, [0, 2000], [0, -300]);
  const rotate1 = useTransform(scrollY, [0, 2000], [0, 180]);
  const rotate2 = useTransform(scrollY, [0, 2000], [0, -120]);

  return (
    <div className="fixed inset-0 z-[-1] overflow-hidden pointer-events-none bg-navy-950 perspective-2000">
      {/* Deep Space Background */}
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-navy-900 via-navy-950 to-black opacity-80" />
      
      {/* Floating Cubes */}
      <GeometricShape 
        type="cube"
        className="w-24 h-24 top-[15%] left-[10%] opacity-20"
        style={{ y: y1, rotateX: rotate1, rotateY: rotate1 }}
      />
      <GeometricShape 
        type="cube"
        className="w-32 h-32 top-[60%] right-[15%] opacity-15"
        style={{ y: y2, rotateX: rotate2, rotateY: rotate2 }}
      />
      
      {/* Floating Orbs */}
      <GeometricShape 
        type="orb"
        className="w-96 h-96 top-[-10%] right-[-5%] opacity-30"
        style={{ y: useTransform(scrollY, [0, 1000], [0, 100]) }}
      />
      <GeometricShape 
        type="orb"
        className="w-[500px] h-[500px] bottom-[-10%] left-[-10%] opacity-20"
        style={{ y: useTransform(scrollY, [0, 1000], [0, -100]) }}
      />

      {/* Grid Floor Effect */}
      <div 
        className="absolute bottom-0 left-0 w-full h-[50vh] opacity-20"
        style={{
          background: 'linear-gradient(to bottom, transparent, #0a0e27), repeating-linear-gradient(90deg, transparent 0, transparent 49px, #d4af37 50px), repeating-linear-gradient(0deg, transparent 0, transparent 49px, #d4af37 50px)',
          transform: 'perspective(1000px) rotateX(60deg) translateY(200px) scale(2)',
        }}
      />
    </div>
  );
};

export default ThreeDBackground;