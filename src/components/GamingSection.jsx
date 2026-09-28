import React from 'react';
import { motion } from 'framer-motion';
import { Gamepad2, Zap, Monitor, ChevronRight } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useNavigate } from 'react-router-dom';

const GamingSection = ({ id }) => {
  const navigate = useNavigate();

  return (
    <section className="py-32 relative overflow-hidden bg-navy-950" id={id}>
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_bottom_right,_rgba(212,175,55,0.1)_0%,_rgba(15,23,42,1)_100%)]" />
      
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        <div className="grid lg:grid-cols-2 gap-16 items-center">
          <motion.div 
            initial={{ opacity: 0, x: -50 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
          >
            <div className="inline-flex items-center rounded-full border border-gold-500/20 bg-gold-500/5 px-4 py-1.5 text-sm font-medium text-gold-400 mb-8">
              <Gamepad2 className="w-4 h-4 mr-2" /> Next-Gen Gaming Engine
            </div>
            
            <h2 className="text-4xl md:text-6xl font-bold text-white mb-6 leading-tight">
              Physics Without <span className="text-transparent bg-clip-text bg-gradient-to-r from-gold-400 to-yellow-200">Compromise.</span>
            </h2>
            
            <p className="text-xl text-gray-400 mb-8 leading-relaxed">
              We are bringing HyperFlux's O(1) computational kernel to AAA gaming. Simulate millions of rigid bodies, fluid dynamics, and complex destructible environments simultaneously with zero frame-drops.
            </p>

            <ul className="space-y-6 mb-10">
              {[
                { icon: Zap, title: "Infinite FPS Physics", desc: "Decouple physics from render loops entirely." },
                { icon: Monitor, title: "Unprecedented Scale", desc: "100x more dynamic objects than current leading engines." }
              ].map((item, i) => (
                <li key={i} className="flex gap-4">
                  <div className="flex-shrink-0 w-12 h-12 rounded-full bg-navy-900 border border-white/10 flex items-center justify-center text-gold-400">
                    <item.icon className="w-6 h-6" />
                  </div>
                  <div>
                    <h4 className="text-white font-bold text-lg">{item.title}</h4>
                    <p className="text-gray-400">{item.desc}</p>
                  </div>
                </li>
              ))}
            </ul>

            <Button
              variant="premium"
              size="lg"
              onClick={() => navigate('/dev-signup')}
              className="h-14 px-8 text-lg group"
            >
              Get Gaming SDK Beta
              <ChevronRight className="w-5 h-5 ml-2 group-hover:translate-x-1 transition-transform" />
            </Button>
          </motion.div>

          <motion.div 
            initial={{ opacity: 0, scale: 0.9 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
            className="relative"
          >
            {/* Abstract visual representing game engine physics */}
            <div className="aspect-square rounded-2xl bg-navy-900 border border-white/10 overflow-hidden relative shadow-luxury-xl flex items-center justify-center p-8">
               <div className="absolute inset-0 bg-[url('https://images.unsplash.com/photo-1552820728-8b83bb6b773f?auto=format&fit=crop&q=80&w=1000')] bg-cover bg-center opacity-30 mix-blend-overlay"></div>
               <div className="absolute inset-0 bg-gradient-to-tr from-navy-950 via-transparent to-gold-500/20"></div>
               
               <div className="relative z-10 grid grid-cols-3 gap-4 w-full h-full">
                  {Array(9).fill(0).map((_, i) => (
                    <motion.div 
                      key={i}
                      animate={{ 
                        y: [0, Math.random() * -40, 0],
                        rotate: [0, Math.random() * 90, 0]
                      }}
                      transition={{ 
                        duration: 3 + Math.random() * 2, 
                        repeat: Infinity,
                        ease: "easeInOut"
                      }}
                      className="bg-white/5 border border-gold-500/30 rounded-lg backdrop-blur-sm"
                    />
                  ))}
               </div>
               
               <div className="absolute top-4 right-4 bg-black/60 backdrop-blur-md px-3 py-1 rounded text-green-400 font-mono text-sm border border-green-500/30">
                 Physics: 999+ FPS
               </div>
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
};

export default GamingSection;