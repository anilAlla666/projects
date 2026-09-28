import React from 'react';
import { motion } from 'framer-motion';
import useScrollToSection from '@/hooks/useScrollToSection';
import { cn } from '@/lib/utils';

const Logo = ({ className }) => {
  const scrollToSection = useScrollToSection();

  const handleLogoClick = (e) => {
    e.preventDefault();
    scrollToSection('hero');
  };

  const nodeVariants = {
    pulse: {
      opacity: [0.7, 1, 0.7],
      scale: [1, 1.2, 1],
      transition: {
        duration: 2,
        repeat: Infinity,
        ease: "easeInOut"
      }
    }
  };

  const containerVariants = {
    initial: { rotateX: 0, rotateY: 0 },
    hover: { 
      rotateX: 5, 
      rotateY: 10,
      transition: { duration: 0.3 }
    }
  };

  return (
    <motion.a
      href="#hero"
      onClick={handleLogoClick}
      className={cn(
        "relative flex items-center gap-3 group cursor-pointer select-none perspective-1000",
        className
      )}
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.6, ease: "easeOut" }}
      whileHover="hover"
    >
      <motion.div
        variants={containerVariants}
        className="relative w-10 h-10 md:w-14 md:h-14 preserve-3d"
      >
        <div className="absolute inset-0 bg-[#FF6B35]/20 blur-xl rounded-full opacity-0 group-hover:opacity-100 transition-opacity duration-300" />

        <svg
          viewBox="0 0 100 100"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
          className="w-full h-full drop-shadow-lg"
        >
          <motion.path
            d="M20,60 L40,30 L70,20 L80,50 L50,80 L20,60 M40,30 L80,50 M20,60 L50,80 M70,20 L50,80"
            stroke="url(#networkGradient)"
            strokeWidth="3"
            strokeLinecap="round"
            strokeLinejoin="round"
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{ duration: 1.5, ease: "easeInOut" }}
          />
          
          <motion.circle cx="20" cy="60" r="6" fill="#FF6B35" variants={nodeVariants} animate="pulse" />
          <motion.circle cx="40" cy="30" r="5" fill="#FF8B35" variants={nodeVariants} animate="pulse" transition={{ delay: 0.2 }} />
          <motion.circle cx="70" cy="20" r="5" fill="#FF6B35" variants={nodeVariants} animate="pulse" transition={{ delay: 0.4 }} />
          <motion.circle cx="80" cy="50" r="6" fill="#FF9B45" variants={nodeVariants} animate="pulse" transition={{ delay: 0.6 }} />
          <motion.circle cx="50" cy="80" r="7" fill="#FF6B35" variants={nodeVariants} animate="pulse" transition={{ delay: 0.8 }} />

          <defs>
            <linearGradient id="networkGradient" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#FF6B35" />
              <stop offset="100%" stopColor="#FF9B45" />
            </linearGradient>
          </defs>
        </svg>
      </motion.div>

      <div className="flex flex-col">
        <span className="font-sora font-extrabold text-sm md:text-lg leading-none tracking-tight text-white group-hover:text-[#FF6B35] transition-colors duration-300">
          NEURAL DYNAMICS
        </span>
        <span className="font-sora font-bold text-xs md:text-sm leading-none tracking-widest text-gray-400 group-hover:text-white transition-colors duration-300">
          TEAM
        </span>
      </div>
    </motion.a>
  );
};

export default Logo;