import React from 'react';
import { motion } from 'framer-motion';
import { Building2, Newspaper, Trophy, Users, Globe, Target, Shield, Zap } from 'lucide-react';
import { CardContainer, CardBody, CardItem } from '@/components/ui/3d-card';

const StatCard = ({ label, value, icon: Icon, subtext }) => (
  <CardContainer className="inter-var w-full">
    <CardBody className="bg-[#18181b] border-[#27272a] w-full rounded-xl p-6 border shadow-xl hover:border-[#f59e0b]/50 transition-colors duration-300">
      <CardItem translateZ="40" className="flex items-center gap-3 mb-4">
        <div className="p-2 rounded-lg bg-[#f59e0b]/10 border border-[#f59e0b]/20">
          <Icon className="w-5 h-5 text-[#f59e0b]" />
        </div>
        <span className="text-gray-400 text-sm font-medium uppercase tracking-wider">{label}</span>
      </CardItem>
      <CardItem translateZ="60">
        <div className="text-3xl font-bold text-[#fbbf24] mb-1">{value}</div>
        <div className="text-xs text-gray-500">{subtext}</div>
      </CardItem>
    </CardBody>
  </CardContainer>
);

const LogoGrid = ({ title, logos }) => (
  <div className="py-8">
    <h3 className="text-sm font-bold text-gray-500 uppercase tracking-widest mb-6 text-center">{title}</h3>
    <div className="grid grid-cols-2 md:grid-cols-4 gap-8 opacity-60">
      {logos.map((logo, i) => (
        <div key={i} className="flex items-center justify-center grayscale hover:grayscale-0 transition-all duration-300 hover:scale-105 cursor-pointer">
          <div className="text-lg font-bold font-sora text-white flex items-center gap-2 group">
            <div className="w-8 h-8 bg-white/10 rounded-full flex items-center justify-center group-hover:text-[#f59e0b] group-hover:bg-[#f59e0b]/10 transition-colors">
              {logo.icon}
            </div>
            {logo.name}
          </div>
        </div>
      ))}
    </div>
  </div>
);

const CredibilitySection = ({ id }) => {
  const values = [
    { name: "Innovation", icon: Zap, desc: "Pushing physics limits" },
    { name: "Excellence", icon: Trophy, desc: "Code quality paramount" },
    { name: "Impact", icon: Globe, desc: "Global scale solutions" },
    { name: "Collaboration", icon: Users, desc: "Open research" },
  ];

  const partners = [
    { name: "Vertex", icon: <Building2 className="w-4 h-4" /> },
    { name: "AI Fund", icon: <Zap className="w-4 h-4" /> },
    { name: "CloudScale", icon: <CloudIcon className="w-4 h-4" /> },
    { name: "NeuralCap", icon: <Shield className="w-4 h-4" /> },
  ];

  const frameworks = [
    { name: "PyTorch", icon: <Target className="w-4 h-4" /> },
    { name: "TensorFlow", icon: <Globe className="w-4 h-4" /> },
    { name: "JAX", icon: <Zap className="w-4 h-4" /> },
    { name: "CUDA", icon: <Shield className="w-4 h-4" /> },
  ];

  const press = [
    { name: "TechCrunch", icon: <Newspaper className="w-4 h-4" /> },
    { name: "Wired", icon: <Globe className="w-4 h-4" /> },
    { name: "Nature", icon: <Building2 className="w-4 h-4" /> },
    { name: "MIT Tech", icon: <Target className="w-4 h-4" /> },
  ];

  return (
    <section className="py-24 relative overflow-hidden border-t border-[#27272a] bg-[#09090b]" id={id}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-24">
          <StatCard 
            label="Funding" 
            value="$10M" 
            subtext="Series A Led by Vertex Ventures" 
            icon={Building2} 
          />
          <StatCard 
            label="Research" 
            value="12" 
            subtext="Published Papers in 2024" 
            icon={Newspaper} 
          />
          <StatCard 
            label="Partners" 
            value="25+" 
            subtext="Enterprise Integrations" 
            icon={Users} 
          />
          <StatCard 
            label="Recognition" 
            value="#1" 
            subtext="Nature Physics Innovation Award" 
            icon={Trophy} 
          />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 mb-24 border-y border-[#27272a] py-12">
          <LogoGrid title="Backed By Industry Leaders" logos={partners} />
          <LogoGrid title="Featured In" logos={press} />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-12">
          <div className="lg:col-span-1">
             <h3 className="text-2xl font-bold text-white mb-6">Our Tech Stack</h3>
             <div className="grid grid-cols-2 gap-4">
                {frameworks.map((tech, i) => (
                  <div key={i} className="flex items-center gap-3 p-4 bg-[#18181b] rounded-lg border border-[#27272a] hover:border-[#f59e0b]/50 transition-colors">
                    <div className="text-[#f59e0b]">{tech.icon}</div>
                    <span className="font-mono text-sm text-gray-300">{tech.name}</span>
                  </div>
                ))}
             </div>
          </div>
          
          <div className="lg:col-span-2">
             <h3 className="text-2xl font-bold text-white mb-6">Core Values</h3>
             <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
               {values.map((value, i) => {
                 const Icon = value.icon;
                 return (
                   <div key={i} className="flex items-start gap-4 p-4 rounded-xl bg-[#18181b] border border-[#27272a] hover:border-[#f59e0b]/30 transition-colors">
                     <div className="p-2 rounded-lg bg-[#f59e0b]/10 text-[#f59e0b]">
                       <Icon className="w-5 h-5" />
                     </div>
                     <div>
                       <h4 className="font-bold text-white mb-1">{value.name}</h4>
                       <p className="text-sm text-gray-400">{value.desc}</p>
                     </div>
                   </div>
                 );
               })}
             </div>
          </div>
        </div>
      </div>
    </section>
  );
};

const CloudIcon = (props) => (
  <svg
    {...props}
    xmlns="http://www.w3.org/2000/svg"
    width="24"
    height="24"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <path d="M17.5 19c0-3.037-2.463-5.5-5.5-5.5S6.5 15.963 6.5 19" />
    <path d="M19 13.5c0-2.485-2.015-4.5-4.5-4.5S10 11.015 10 13.5" />
    <path d="M5 19h14" />
  </svg>
);

export default CredibilitySection;