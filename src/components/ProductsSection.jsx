
import React from 'react';
import { motion } from 'framer-motion';
import ProductCard from './ProductCard';

const products = [
  {
    id: 'cipher',
    name: 'CIPHER',
    description: "A persistent execution layer that exists between every application ever written for NVIDIA GPUs and the silicon those applications run on. It installs with one environment variable. It requires zero code changes. The application above it does not know CIPHER exists. The GPU below it does not know CIPHER exists. And yet every single kernel launch passes through it.",
    features: [
      "7.76× peak kernel speedup",
      "694 TFLOPS sustained",
      "Zero numerical difference on validated operations (max_diff = 0.000000)",
      "Zero code changes required",
      "One environment variable deployment",
      "Intercepts every kernel launch automatically"
    ],
    technicalHighlights: "Koopman operation decomposition, L2 cache optimization, neural network prediction (CfC), 31 observer operations"
  },
  {
    id: 'tirusoma',
    name: 'TIRUSOMA',
    description: "Advanced orchestration and resource management platform for distributed GPU clusters. Maximizes utilization and minimizes idle time across diverse and complex distributed workloads.",
    features: [
      "Dynamic resource allocation in real-time",
      "Sub-millisecond latency orchestration",
      "Seamless scaling across 10,000+ nodes",
      "Fault-tolerant distributed execution",
      "Hardware-agnostic abstraction layer",
      "Advanced telemetry and cluster profiling"
    ],
    technicalHighlights: "Distributed hash table state management, Kernel-level network bypass, Predictive workload scheduling, Zero-copy memory transfers"
  }
];

export default function ProductsSection() {
  return (
    <section id="products" className="py-24 bg-background text-foreground relative z-10">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="max-w-3xl mx-auto text-center mb-16">
          <motion.h2
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-3xl md:text-4xl font-bold tracking-tight mb-4"
          >
            Our Core Technologies
          </motion.h2>
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.1 }}
            className="text-lg text-muted-foreground"
          >
            Redefining performance boundaries with software-defined hardware acceleration.
          </motion.p>
        </div>

        <div className="grid md:grid-cols-2 gap-8 max-w-6xl mx-auto">
          {products.map((product) => (
            <ProductCard key={product.id} product={product} />
          ))}
        </div>
      </div>
    </section>
  );
}
