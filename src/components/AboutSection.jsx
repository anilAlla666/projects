
import React from 'react';
import { motion } from 'framer-motion';

const AboutSection = ({ id }) => {
  return (
    <section id={id} className="py-24 bg-secondary/50">
      <div className="container mx-auto px-6 max-w-4xl text-center">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          <h2 className="text-3xl md:text-4xl font-bold mb-8 tracking-tight text-foreground">
            About the Founder
          </h2>
          <div className="w-20 h-1 bg-primary mx-auto mb-8 rounded-full" />
          <p className="text-lg md:text-xl leading-relaxed text-muted-foreground">
            <strong className="text-foreground font-semibold">Anil Kumar Alla</strong> is the founder of Neural Dynamics Team. With a profound background in advanced algorithm optimization and neural inference architecture, his vision is to replace traditional iterative solvers with constant-time inference models. By fundamentally shifting the computing paradigm from O(N) to O(1), he is driving unprecedented performance breakthroughs across complex domains like gaming, robotics, and real-time physical simulation.
          </p>
        </motion.div>
      </div>
    </section>
  );
};

export default AboutSection;
