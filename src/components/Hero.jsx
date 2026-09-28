
import React from 'react';
import { motion } from 'framer-motion';
import { Button } from '@/components/ui/button';
import { Mail, ArrowRight } from 'lucide-react';

const Hero = ({ id }) => {
  return (
    <section 
      id={id}
      className="relative min-h-[90vh] flex items-center justify-center pt-20 overflow-hidden bg-black"
    >
      {/* Background Image & Overlay */}
      <div className="absolute inset-0 z-0">
        <img 
          src="https://images.unsplash.com/photo-1677442136019-21780ecad995" 
          alt="Abstract neural network computing concept" 
          className="w-full h-full object-cover opacity-60"
        />
        <div className="absolute inset-0 bg-gradient-to-b from-black/80 via-black/60 to-background/95" />
      </div>

      <div className="container relative z-10 mx-auto px-6 max-w-5xl text-center">
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="mb-8 inline-flex items-center rounded-full border border-white/20 bg-white/10 px-6 py-2 text-sm font-medium tracking-wide text-white backdrop-blur-sm"
        >
          Neural Dynamics Team
        </motion.div>

        <div className="space-y-4 mb-12">
          <motion.h1 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="text-4xl md:text-6xl lg:text-7xl font-bold tracking-tight text-white leading-tight"
          >
            Every iterative algorithm.<br />
            Constant time.<br />
            One forward pass.
          </motion.h1>
        </div>

        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="text-lg md:text-xl text-gray-300 max-w-3xl mx-auto mb-12 font-light"
        >
          We build architectures that fundamentally shift complex system performance from O(N) to O(1).
        </motion.p>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.6 }}
          className="flex flex-col sm:flex-row gap-4 justify-center items-center"
        >
          <a href="mailto:anilkumaralla@neuraldynamicsteam.io" className="w-full sm:w-auto">
            <Button size="lg" className="w-full sm:w-auto bg-white text-black hover:bg-gray-200 text-base px-8 h-14 rounded-full font-medium">
              Get Access
              <ArrowRight className="w-5 h-5 ml-2" />
            </Button>
          </a>
          <a href="mailto:anilkumaralla@neuraldynamicsteam.io" className="w-full sm:w-auto">
            <Button size="lg" variant="outline" className="w-full sm:w-auto border-white/30 text-white hover:bg-white/10 text-base px-8 h-14 rounded-full font-medium bg-transparent">
              <Mail className="w-5 h-5 mr-2" />
              Contact Us
            </Button>
          </a>
        </motion.div>
      </div>
    </section>
  );
};

export default Hero;
