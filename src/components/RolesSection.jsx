import React from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { Briefcase, ChevronRight } from 'lucide-react';
import { Button } from '@/components/ui/button';

const RolesSection = () => {
  const navigate = useNavigate();

  return (
    <section className="py-24 relative overflow-hidden bg-[#09090b]">
      <div className="absolute inset-0 bg-gradient-to-b from-[#09090b] via-[#18181b]/40 to-[#09090b] pointer-events-none" />
      
      <div className="container mx-auto px-4 relative z-10">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8 }}
          className="max-w-4xl mx-auto"
        >
          <div className="bg-[#18181b] border border-[#27272a] hover:border-[#f59e0b]/30 transition-colors p-10 md:p-14 rounded-3xl text-center relative overflow-hidden group">
            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-[#f59e0b]/5 to-transparent translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-1000 ease-in-out pointer-events-none" />

            <div className="inline-flex items-center justify-center p-3 bg-[#f59e0b]/10 rounded-xl mb-6 text-[#f59e0b]">
              <Briefcase className="w-8 h-8" />
            </div>

            <h2 className="text-4xl md:text-5xl font-bold text-white mb-6">
              Join Our <span className="text-[#f59e0b]">Team</span>
            </h2>

            <p className="text-gray-300 text-lg md:text-xl mb-10 max-w-2xl mx-auto leading-relaxed">
              We are assembling a specialized unit of engineers, researchers, and visionaries within the Neural Dynamics Team to redefine the boundaries of computational physics. Help us build the O(1) kernel.
            </p>

            <motion.div
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <Button
                onClick={() => navigate('/roles')}
                className="bg-gradient-to-r from-[#f59e0b] to-[#d97706] hover:from-[#fbbf24] hover:to-[#f59e0b] text-black font-bold border-0 h-14 px-10 text-lg rounded-full shadow-lg hover:shadow-[#f59e0b]/25 transition-all duration-300 group/btn"
              >
                View Open Roles
                <ChevronRight className="w-5 h-5 ml-2 group-hover/btn:translate-x-1 transition-transform" />
              </Button>
            </motion.div>
          </div>
        </motion.div>
      </div>
    </section>
  );
};

export default RolesSection;