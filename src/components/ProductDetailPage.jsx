
import React, { useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { ArrowLeft, CheckCircle2, Cpu, Zap, BarChart, Server } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useToast } from '@/components/ui/use-toast';
import Navbar from './Navbar';
import Footer from './Footer';

const productsData = {
  cipher: {
    name: 'CIPHER',
    tagline: 'The universal execution layer for NVIDIA GPUs.',
    description: "A persistent execution layer that exists between every application ever written for NVIDIA GPUs and the silicon those applications run on. It installs with one environment variable. It requires zero code changes. The application above it does not know CIPHER exists. The GPU below it does not know CIPHER exists. And yet every single kernel launch passes through it.",
    features: [
      "7.76× peak kernel speedup",
      "694 TFLOPS sustained",
      "Zero numerical difference on validated operations (max_diff = 0.000000)",
      "Zero code changes required",
      "One environment variable deployment",
      "Intercepts every kernel launch automatically"
    ],
    technicalHighlights: "Koopman operation decomposition, L2 cache optimization, neural network prediction (CfC), 31 observer operations",
    icon: <Cpu className="w-12 h-12" />,
    specs: [
      { label: "Deployment", value: "1 Environment Variable" },
      { label: "Code Changes", value: "Zero" },
      { label: "Peak Speedup", value: "7.76x" },
      { label: "Numerical Diff", value: "0.000000" }
    ]
  },
  tirusoma: {
    name: 'TIRUSOMA',
    tagline: 'Next-generation distributed GPU orchestration.',
    description: "Advanced orchestration and resource management platform for distributed GPU clusters. Maximizes utilization and minimizes idle time across diverse and complex distributed workloads.",
    features: [
      "Dynamic resource allocation in real-time",
      "Sub-millisecond latency orchestration",
      "Seamless scaling across 10,000+ nodes",
      "Fault-tolerant distributed execution",
      "Hardware-agnostic abstraction layer",
      "Advanced telemetry and cluster profiling"
    ],
    technicalHighlights: "Distributed hash table state management, Kernel-level network bypass, Predictive workload scheduling, Zero-copy memory transfers",
    icon: <Server className="w-12 h-12" />,
    specs: [
      { label: "Latency", value: "< 1ms" },
      { label: "Max Nodes", value: "10,000+" },
      { label: "Scheduling", value: "Predictive" },
      { label: "State Management", value: "Distributed Hash Table" }
    ]
  }
};

export default function ProductDetailPage() {
  const { productId } = useParams();
  const navigate = useNavigate();
  const { toast } = useToast();
  const product = productsData[productId];

  useEffect(() => {
    window.scrollTo(0, 0);
  }, [productId]);

  if (!product) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center bg-background text-foreground">
        <h1 className="text-4xl font-bold mb-4">Product Not Found</h1>
        <Button onClick={() => navigate('/')}>Return Home</Button>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col bg-background text-foreground">
      <Navbar />

      <main className="flex-grow pt-24 pb-16">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <Button
            variant="ghost"
            onClick={() => navigate('/')}
            className="mb-8"
          >
            <ArrowLeft className="w-4 h-4 mr-2" /> Back to Products
          </Button>

          <div className="grid lg:grid-cols-2 gap-16 items-start">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5 }}
            >
              <div className="inline-flex items-center justify-center p-4 bg-primary/10 text-primary rounded-2xl mb-6">
                {product.icon}
              </div>
              <h1 className="text-4xl sm:text-5xl font-extrabold tracking-tight mb-4">
                {product.name}
              </h1>
              <p className="text-xl text-primary font-medium mb-6">
                {product.tagline}
              </p>
              <p className="text-lg text-muted-foreground leading-relaxed mb-8">
                {product.description}
              </p>

              <div className="grid grid-cols-2 gap-4 mb-8">
                {product.specs.map((spec, idx) => (
                  <div key={idx} className="bg-card border border-border rounded-xl p-4 shadow-sm">
                    <p className="text-sm text-muted-foreground mb-1">{spec.label}</p>
                    <p className="font-semibold text-lg">{spec.value}</p>
                  </div>
                ))}
              </div>

              <Button 
                size="lg" 
                className="w-full sm:w-auto"
                onClick={() => toast({ description: "🚧 This feature isn't implemented yet—but don't worry! You can request it in your next prompt! 🚀" })}
              >
                Request Demo
              </Button>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="space-y-8"
            >
              <div className="bg-card border border-border rounded-2xl p-8 shadow-sm">
                <h3 className="text-2xl font-bold mb-6 flex items-center gap-3">
                  <Zap className="text-primary w-6 h-6" /> Key Features
                </h3>
                <ul className="space-y-4">
                  {product.features.map((feature, idx) => (
                    <li key={idx} className="flex items-start gap-3">
                      <CheckCircle2 className="w-6 h-6 text-primary shrink-0" />
                      <span className="text-muted-foreground">{feature}</span>
                    </li>
                  ))}
                </ul>
              </div>

              <div className="bg-primary/5 border border-primary/20 rounded-2xl p-8">
                <h3 className="text-2xl font-bold mb-4 flex items-center gap-3">
                  <BarChart className="text-primary w-6 h-6" /> Technical Highlights
                </h3>
                <p className="text-muted-foreground leading-relaxed">
                  {product.technicalHighlights}
                </p>
              </div>
            </motion.div>
          </div>
        </div>
      </main>

      <Footer />
    </div>
  );
}
