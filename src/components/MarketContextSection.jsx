import React from 'react';

const MarketContextSection = () => {
  return (
    <section className="py-24 bg-[#18181b] border-y border-[#27272a]">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="mb-12">
          <h2 className="text-3xl md:text-4xl font-serif font-normal text-white mb-2">Market Context</h2>
          <p className="text-xl font-sans text-[#f59e0b]">Replacing code, not robots.</p>
        </div>
        
        <div className="space-y-6 font-sans text-gray-300 text-lg leading-relaxed">
          <p>
            The humanoid robotics market is expanding rapidly, with hardware manufacturers like Figure AI, Boston Dynamics, and Unitree pushing physical capabilities to the limit. However, the software bottleneck remains: brittle, hand-coded control systems or computationally massive reinforcement learning models.
          </p>
          <p>
            Current industry leaders spend months manually tuning Model Predictive Control (MPC) parameters for every new robot or task. If a robot is modified by even a few kilograms, the entire control stack must be re-engineered.
          </p>
          <p>
            While companies like Skild AI focus on massive foundational models for high-level reasoning, Neural Dynamics solves the foundational <em className="italic">low-level</em> problem: the motor cortex.
          </p>
          <p>
            TIRUSOMA acts as the universal OS for physical actuation, abstracting away hardware complexities. We are not building robots; we are licensing the essential, highly efficient software cerebellum that makes next-generation robotics possible.
          </p>
        </div>
      </div>
    </section>
  );
};

export default MarketContextSection;