import React from 'react';
import { motion } from 'framer-motion';
import { Cpu, Network, Zap, Lock, Globe, Code } from 'lucide-react';

const TechCube = ({ icon: Icon, title, description, color }) => {
  return (
    <div className="w-full h-64 perspective-1000 group cursor-pointer">
      <div className="relative w-full h-full duration-700 transform-3d group-hover:rotate-y-180">
        
        {/* Front Face */}
        <div className="absolute inset-0 w-full h-full backface-hidden bg-navy-900/60 backdrop-blur-xl border border-white/10 rounded-2xl p-8 flex flex-col items-center justify-center text-center shadow-luxury">
          <div className={`w-16 h-16 rounded-full bg-white/5 flex items-center justify-center mb-6 ${color}`}>
            <Icon className="w-8 h-8" />
          </div>
          <h3 className="text-xl font-bold text-white mb-2">{title}</h3>
          <p className="text-sm text-gray-400">Hover to explore architecture</p>
        </div>

        {/* Back Face */}
        <div className="absolute inset-0 w-full h-full backface-hidden rotate-y-180 bg-gradient-to-br from-navy-800 to-navy-950 border border-gold-500/30 rounded-2xl p-8 flex flex-col items-center justify-center text-center shadow-luxury-lg">
          <h3 className="text-lg font-bold text-gold-400 mb-4">{title}</h3>
          <p className="text-sm text-gray-300 leading-relaxed">
            {description}
          </p>
          <div className="mt-6 w-full h-1 bg-navy-700 rounded-full overflow-hidden">
            <div className="h-full bg-gold-500 w-2/3 animate-pulse" />
          </div>
        </div>

      </div>
    </div>
  );
};

const TechnologyGrid = ({ id }) => {
  const technologies = [
    { icon: Cpu, title: "Neural Core", description: "Liquid neural networks adapting in real-time to solution manifolds.", color: "text-blue-400" },
    { icon: Network, title: "Mesh Logic", description: "Self-optimizing topology that eliminates redundant computations.", color: "text-purple-400" },
    { icon: Zap, title: "Instant Solve", description: "Direct mapping from problem space to solution space via phase encoding.", color: "text-yellow-400" },
    { icon: Lock, title: "Crypto Guard", description: "Homomorphic encryption enabling secure processing on untrusted nodes.", color: "text-green-400" },
    { icon: Globe, title: "Global State", description: "Distributed state management with eventual consistency guarantees.", color: "text-cyan-400" },
    { icon: Code, title: "Native API", description: "Drop-in replacement for standard BLAS/LAPACK libraries.", color: "text-red-400" },
  ];

  return (
    <section className="py-32 relative" id={id}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div 
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-20"
        >
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-6">
            The <span className="gradient-text">Core Stack</span>
          </h2>
          <p className="text-gray-400 max-w-2xl mx-auto">
            Explore the multi-dimensional architecture powering the HyperFlux kernel.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {technologies.map((tech, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, scale: 0.9 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.1 }}
            >
              <TechCube {...tech} />
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default TechnologyGrid;