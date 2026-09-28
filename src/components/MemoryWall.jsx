import React from 'react';
import { motion, useScroll, useTransform } from 'framer-motion';
import { TrendingUp, Zap, Database, Activity } from 'lucide-react';
import { CardContainer, CardBody, CardItem } from './ui/3d-card';

const MemoryWall = ({ id }) => {
  const { scrollYProgress } = useScroll();
  const rotateX = useTransform(scrollYProgress, [0, 1], [20, -20]);

  const stats = [
    { label: "Throughput", value: "9.4k", unit: "x", desc: "Faster than CUDA", icon: Zap },
    { label: "Latency", value: "<1", unit: "ms", desc: "Constant time", icon: TrendingUp },
    { label: "Memory", value: "O(1)", unit: "", desc: "Invariant scaling", icon: Database },
    { label: "Efficiency", value: "99.9", unit: "%", desc: "Resource utilization", icon: Activity },
  ];

  return (
    <section className="py-32 relative preserve-3d perspective-2000 bg-[#09090b]" id={id}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div 
          className="text-center mb-20 preserve-3d"
          style={{ rotateX }}
        >
          <h2 className="text-4xl md:text-6xl font-serif font-normal text-white mb-6">
            Beyond the <span className="text-[#f59e0b] font-serif">Memory Wall</span>
          </h2>
          <p className="text-xl font-sans text-gray-400 max-w-2xl mx-auto">
            Traditional architectures are bottlenecked by data movement. HyperFlux eliminates the bottleneck.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          {stats.map((stat, index) => {
            const Icon = stat.icon;
            return (
              <CardContainer key={index} className="inter-var w-full h-full">
                <CardBody className="bg-[#18181b] relative group/card border-[#27272a] w-full h-auto rounded-xl p-8 border hover:border-[#f59e0b]/50 shadow-xl transition-colors duration-300">
                  <CardItem translateZ="50" className="w-full flex justify-between items-start mb-6">
                    <div className={`p-3 rounded-lg bg-[#f59e0b]/10 border border-[#f59e0b]/20 text-[#f59e0b]`}>
                      <Icon className="w-6 h-6" />
                    </div>
                    <div className="text-xs font-sans font-semibold text-[#f59e0b] uppercase tracking-widest border border-[#f59e0b]/30 px-2 py-1 rounded bg-[#f59e0b]/5">
                      Verified
                    </div>
                  </CardItem>
                  
                  <CardItem translateZ="80" className="mb-2">
                    <div className="flex items-baseline gap-1">
                      <span className="text-5xl font-mono font-bold text-[#fbbf24] tracking-tighter">{stat.value}</span>
                      <span className={`text-xl font-mono font-bold text-[#f59e0b]`}>{stat.unit}</span>
                    </div>
                  </CardItem>
                  
                  <CardItem translateZ="60" className="text-lg font-serif font-normal text-gray-300 mb-1">
                    {stat.label}
                  </CardItem>
                  
                  <CardItem translateZ="40" className="text-sm font-sans text-gray-500">
                    {stat.desc}
                  </CardItem>

                  {/* 3D Depth Layer */}
                  <div className="absolute inset-0 bg-gradient-to-br from-[#f59e0b]/5 to-transparent rounded-xl opacity-0 group-hover/card:opacity-100 transition-opacity pointer-events-none transform translate-z-[-20px]" />
                </CardBody>
              </CardContainer>
            );
          })}
        </div>
      </div>
    </section>
  );
};

export default MemoryWall;