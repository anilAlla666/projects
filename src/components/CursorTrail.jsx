import React from 'react';
import { motion } from 'framer-motion';
import useCursorTrail from '@/hooks/useCursorTrail';

const CursorTrail = () => {
  const { position, isTouchDevice } = useCursorTrail();

  if (isTouchDevice) return null;

  return (
    <>
      <motion.div
        className="fixed top-0 left-0 w-2 h-2 bg-[#f59e0b] rounded-full pointer-events-none z-[100]"
        animate={{ x: position.x - 4, y: position.y - 4 }}
        transition={{ type: "spring", stiffness: 500, damping: 28, mass: 2 }}
      />
      <motion.div
        className="fixed top-0 left-0 w-8 h-8 border border-[#f59e0b]/50 rounded-full pointer-events-none z-[99]"
        animate={{ x: position.x - 16, y: position.y - 16 }}
        transition={{ type: "spring", stiffness: 250, damping: 20, mass: 1 }}
      />
    </>
  );
};

export default CursorTrail;