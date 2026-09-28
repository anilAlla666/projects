import React from 'react';
import { motion } from 'framer-motion';
import ScrollAnimatedSection from '@/components/ScrollAnimatedSection';

const ComparisonTable = () => {
  const rows = [
    { prop: "Deployment size", cpp: "10-50 MB", mpc: "100+ MB", rl: "1-5 MB", tirusoma: "244 KB" },
    { prop: "Control frequency", cpp: "500 Hz", mpc: "100-200 Hz", rl: "50 Hz (Varies)", tirusoma: "10 kHz+" },
    { prop: "New robot setup", cpp: "Months of tuning", mpc: "Heavy math remodel", rl: "Days of retraining", tirusoma: "Minutes (Zero-shot)" },
    { prop: "New behavior design", cpp: "Hardcoded state machines", mpc: "Complex cost functions", rl: "Reward engineering", tirusoma: "Latent space prompt" },
    { prop: "Safety certification", cpp: "Standard manual checks", mpc: "Standard constraints", rl: "Impossible (Black box)", tirusoma: "Native (CBF Filter)" },
    { prop: "Hardware required", cpp: "CPU", mpc: "High-end CPU/GPU", rl: "GPU Required", tirusoma: "Standard MCU" },
    { prop: "Timing determinism", cpp: "Variable execution", mpc: "O(N³) Variable", rl: "Variable latency", tirusoma: "O(1) Strict bounds" },
  ];

  return (
    <section className="py-24 bg-background">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <ScrollAnimatedSection>
          <h2 className="text-3xl font-serif font-normal text-[#fafafa] mb-10 text-center">Architectural Comparison</h2>
        </ScrollAnimatedSection>
        
        <div className="w-full overflow-x-auto rounded-xl border border-border bg-card shadow-xl">
          <table className="w-full text-left border-collapse min-w-[800px]">
            <thead>
              <tr className="border-b border-border bg-background/50">
                <th className="p-6 text-sm font-semibold text-[#a1a1aa] uppercase tracking-wider w-1/5">Property</th>
                <th className="p-6 text-sm font-semibold text-[#a1a1aa] uppercase tracking-wider w-1/5">Hand-Coded C++</th>
                <th className="p-6 text-sm font-semibold text-[#a1a1aa] uppercase tracking-wider w-1/5">MPC</th>
                <th className="p-6 text-sm font-semibold text-[#a1a1aa] uppercase tracking-wider w-1/5">RL Policy (MLP)</th>
                <th className="p-6 text-sm font-bold text-[#fafafa] uppercase tracking-wider bg-[#f59e0b]/20 w-1/5">TIRUSOMA</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {rows.map((row, i) => (
                <motion.tr 
                  key={i}
                  initial={{ opacity: 0, y: 10 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true, margin: "-50px" }}
                  transition={{ delay: i * 0.08, duration: 0.4 }}
                  className="hover:bg-muted/50 transition-colors group"
                >
                  <td className="p-6 text-sm font-medium text-[#fafafa]">{row.prop}</td>
                  <td className="p-6 text-sm text-[#a1a1aa]">{row.cpp}</td>
                  <td className="p-6 text-sm text-[#a1a1aa]">{row.mpc}</td>
                  <td className="p-6 text-sm text-[#a1a1aa]">{row.rl}</td>
                  <td className="p-6 text-sm font-bold text-[#fafafa] bg-[#f59e0b]/10 group-hover:bg-[#f59e0b]/20 transition-colors">{row.tirusoma}</td>
                </motion.tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
};

export default ComparisonTable;