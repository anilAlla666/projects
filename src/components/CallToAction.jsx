import React from 'react';
import { motion } from 'framer-motion';
import { Button } from '@/components/ui/button';
import { ArrowRight, Box } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

const CallToAction = ({ id }) => {
  const navigate = useNavigate();

  return (
    <section className="py-32 relative overflow-hidden perspective-1000 bg-[#09090b]" id={id}>
       <div className="absolute inset-0 bg-gradient-to-b from-[#09090b] to-[#18181b]" />
       
       <motion.div 
         className="absolute top-10 right-10 w-32 h-32 border border-[#f59e0b]/20 rounded-full"
         animate={{ y: [0, -30, 0], rotateX: [0, 45, 0] }}
         transition={{ duration: 8, repeat: Infinity }}
       />

       <div className="max-w-5xl mx-auto px-4 relative z-10 text-center">
         <motion.div
           initial={{ scale: 0.9, rotateX: 10, opacity: 0 }}
           whileInView={{ scale: 1, rotateX: 0, opacity: 1 }}
           viewport={{ once: true }}
           transition={{ duration: 0.8 }}
           className="bg-[#18181b] border border-[#27272a] hover:border-[#f59e0b]/30 transition-colors p-16 rounded-3xl"
         >
           <h2 className="text-4xl md:text-6xl font-bold text-white mb-8">
             Ready to <span className="text-[#f59e0b]">Ascend?</span>
           </h2>
           <p className="text-xl text-gray-300 mb-10 max-w-2xl mx-auto">
             Join the closed beta and deploy your first O(1) kernel today with the Neural Dynamics Team.
           </p>
           
           <div className="flex flex-col sm:flex-row justify-center gap-6">
             <Button 
               size="lg" 
               onClick={() => navigate('/dev-signup')}
               className="h-16 px-10 text-lg font-bold rounded-xl shadow-[0_10px_40px_-10px_rgba(245,158,11,0.5)] hover:shadow-[0_20px_60px_-10px_rgba(245,158,11,0.6)] hover:translate-y-[-4px] transition-all"
             >
               Explore Our Work <ArrowRight className="ml-2" />
             </Button>
             
             <Button 
               variant="outline" 
               size="lg" 
               onClick={() => window.open('https://careers.hyperflux.ai', '_blank')}
               className="h-16 px-10 text-lg border-[#27272a] text-white hover:border-[#f59e0b]/50 hover:bg-[#f59e0b]/10 hover:text-[#fbbf24] transition-colors"
             >
               View Careers <Box className="ml-2 w-4 h-4" />
             </Button>
           </div>
         </motion.div>
       </div>
    </section>
  );
};

export default CallToAction;