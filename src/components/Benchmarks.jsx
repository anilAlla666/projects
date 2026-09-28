import React from 'react';
import { motion, useScroll, useTransform } from 'framer-motion';
import { CardContainer, CardBody, CardItem } from './ui/3d-card';
import { Activity, BarChart3, Zap } from 'lucide-react';

const Benchmarks = ({ id }) => {
  const { scrollYProgress } = useScroll();
  const scale = useTransform(scrollYProgress, [0, 1], [0.95, 1.05]);

  const benchmarks = [
    { title: "Fluid Dynamics", score: "9,840", unit: "FPS", baseline: "60 FPS (CUDA)", improvement: "+16,300%", icon: Activity },
    { title: "N-Body Sim", score: "0.02", unit: "ms", baseline: "14.5 ms (AVX-512)", improvement: "+72,400%", icon: Zap },
    { title: "Weather Model", score: "Real-time", unit: "", baseline: "4 hours (Cluster)", improvement: "Instant", icon: BarChart3 },
  ];

  return (
    <section className="py-32 relative perspective-1000 bg-[#09090b]" id={id}>
      <motion.div 
        className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8"
        style={{ scale }}
      >
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-serif font-normal text-white mb-6">
            Verified Benchmarks
          </h2>
          <p className="text-gray-400 font-sans">Validated on standard EPYC and A100 clusters.</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {benchmarks.map((bench, index) => {
            const Icon = bench.icon;
            return (
              <CardContainer key={index} className="inter-var w-full h-full">
                <CardBody className="bg-[#18181b] relative group/card border-[#27272a] w-full h-auto rounded-xl p-8 border hover:border-[#f59e0b]/50 shadow-xl transition-colors duration-300">
                  
                  {/* Floating Number Effect */}
                  <CardItem translateZ="100" className="w-full text-center mb-8">
                    <div className="inline-block relative">
                       <span className="text-6xl font-mono font-bold text-[#fbbf24]">
                         {bench.score}
                       </span>
                       <span className="text-xl font-mono text-[#f59e0b] font-bold ml-2">{bench.unit}</span>
                       <div className="absolute -inset-4 bg-[#f59e0b]/20 blur-xl rounded-full -z-10 animate-pulse"></div>
                    </div>
                  </CardItem>

                  <CardItem translateZ="60" className="flex items-center justify-center gap-3 mb-6">
                    <Icon className="w-5 h-5 text-[#f59e0b]" />
                    <h3 className="text-xl font-serif font-normal text-white">{bench.title}</h3>
                  </CardItem>

                  <CardItem translateZ="40" className="w-full bg-[#09090b]/50 rounded-lg p-4 border border-[#27272a]">
                    <div className="flex justify-between font-sans text-sm text-gray-400 mb-2">
                      <span>Baseline:</span>
                      <span className="font-mono">{bench.baseline}</span>
                    </div>
                    <div className="flex justify-between font-sans text-sm font-semibold text-[#f59e0b]">
                      <span>Improvement:</span>
                      <span className="font-mono font-bold">{bench.improvement}</span>
                    </div>
                  </CardItem>

                </CardBody>
              </CardContainer>
            );
          })}
        </div>
      </motion.div>
    </section>
  );
};

export default Benchmarks;