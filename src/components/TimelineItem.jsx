import React from 'react';
import { CardContainer, CardBody, CardItem } from '@/components/ui/3d-card';
import { motion } from 'framer-motion';

const TimelineItem = ({ phase, title, description, icon: Icon, isLast }) => {
  return (
    <div className="relative pl-8 md:pl-0">
      {/* Mobile Timeline Line */}
      <div className="absolute left-0 top-0 bottom-0 w-px bg-gradient-to-b from-[#f59e0b]/50 to-transparent md:hidden" />
      
      {/* Desktop Timeline Line connection handled in parent grid */}
      
      <CardContainer className="inter-var w-full">
        <CardBody className="bg-[#18181b] relative group/card border-[#27272a] w-full h-auto rounded-xl p-8 border hover:border-[#f59e0b]/50 shadow-xl transition-colors duration-300">
          <CardItem translateZ="50" className="w-full flex justify-between items-start mb-6">
            <div className="p-3 rounded-lg bg-[#f59e0b]/10 border border-[#f59e0b]/20 text-[#f59e0b]">
              <Icon className="w-6 h-6" />
            </div>
            <div className="px-3 py-1 rounded-full border border-[#f59e0b]/30 bg-[#f59e0b]/10 text-[#fbbf24] text-xs font-sans font-bold uppercase tracking-wider">
              Phase {phase}
            </div>
          </CardItem>
          
          <CardItem translateZ="60" className="text-xl font-serif font-normal text-white mb-3">
            {title}
          </CardItem>
          
          <CardItem translateZ="40" className="text-sm font-sans text-gray-400 leading-relaxed">
            {description}
          </CardItem>
        </CardBody>
      </CardContainer>
    </div>
  );
};

export default TimelineItem;