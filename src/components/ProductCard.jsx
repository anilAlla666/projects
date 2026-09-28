
import React from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { ChevronRight, Cpu, Zap, Activity } from 'lucide-react';

export default function ProductCard({ product }) {
  const navigate = useNavigate();

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.5 }}
      className="bg-card text-card-foreground rounded-xl border border-border overflow-hidden flex flex-col h-full hover:shadow-lg transition-shadow"
    >
      <div className="p-8 flex flex-col flex-grow">
        <div className="flex items-center gap-3 mb-4">
          <div className="p-3 bg-primary/10 rounded-lg text-primary">
            <Cpu className="w-6 h-6" />
          </div>
          <h3 className="text-2xl font-bold tracking-tight">{product.name}</h3>
        </div>

        <p className="text-muted-foreground mb-6 flex-grow leading-relaxed">
          {product.description}
        </p>

        <div className="space-y-6 mb-8">
          <div>
            <h4 className="font-semibold mb-3 flex items-center gap-2">
              <Zap className="w-4 h-4 text-primary" /> Key Features
            </h4>
            <ul className="space-y-2">
              {product.features.map((feature, idx) => (
                <li key={idx} className="flex items-start gap-2 text-sm text-muted-foreground">
                  <div className="w-1.5 h-1.5 rounded-full bg-primary mt-1.5 shrink-0" />
                  <span>{feature}</span>
                </li>
              ))}
            </ul>
          </div>

          <div>
            <h4 className="font-semibold mb-3 flex items-center gap-2">
              <Activity className="w-4 h-4 text-primary" /> Technical Highlights
            </h4>
            <p className="text-sm text-muted-foreground bg-muted p-4 rounded-lg">
              {product.technicalHighlights}
            </p>
          </div>
        </div>

        <Button
          onClick={() => navigate(`/product/${product.id}`)}
          className="w-full mt-auto group"
          size="lg"
        >
          Learn More
          <ChevronRight className="w-4 h-4 ml-2 group-hover:translate-x-1 transition-transform" />
        </Button>
      </div>
    </motion.div>
  );
}
