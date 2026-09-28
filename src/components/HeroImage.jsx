import React, { useEffect, useRef } from 'react';
import { motion } from 'framer-motion';

const HeroImage = () => {
  const canvasRef = useRef(null);
  const containerRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const ctx = canvas.getContext('2d');
    let animationFrameId;
    let width, height;

    // Configuration
    const config = {
      nodeCount: 40,
      connectionDistance: 150,
      colors: {
        navy: '#0f172a',
        gold: '#fbbf24',
        red: '#dc2626',
        nodes: ['#fbbf24', '#dc2626', '#ffffff'] // Gold, Red, White
      },
      mouseInteractionRadius: 200
    };

    // State
    const nodes = [];
    let mouseX = -1000;
    let mouseY = -1000;

    // Classes
    class Node {
      constructor() {
        this.x = Math.random() * width;
        this.y = Math.random() * height;
        this.vx = (Math.random() - 0.5) * 0.5;
        this.vy = (Math.random() - 0.5) * 0.5;
        this.radius = Math.random() * 2 + 1.5;
        this.color = config.colors.nodes[Math.floor(Math.random() * config.colors.nodes.length)];
        this.pulsePhase = Math.random() * Math.PI * 2;
        this.pulseSpeed = 0.05 + Math.random() * 0.05;
      }

      update() {
        // Movement
        this.x += this.vx;
        this.y += this.vy;

        // Bounce off walls
        if (this.x < 0 || this.x > width) this.vx *= -1;
        if (this.y < 0 || this.y > height) this.vy *= -1;

        // Mouse interaction
        const dx = mouseX - this.x;
        const dy = mouseY - this.y;
        const distance = Math.sqrt(dx * dx + dy * dy);

        if (distance < config.mouseInteractionRadius) {
          const forceDirectionX = dx / distance;
          const forceDirectionY = dy / distance;
          const force = (config.mouseInteractionRadius - distance) / config.mouseInteractionRadius;
          const directionX = forceDirectionX * force * 0.5;
          const directionY = forceDirectionY * force * 0.5;
          this.vx -= directionX;
          this.vy -= directionY;
        }

        // Limit speed
        const speed = Math.sqrt(this.vx * this.vx + this.vy * this.vy);
        if (speed > 2) {
          this.vx = (this.vx / speed) * 2;
          this.vy = (this.vy / speed) * 2;
        }

        // Pulse
        this.pulsePhase += this.pulseSpeed;
      }

      draw() {
        const pulse = (Math.sin(this.pulsePhase) + 1) / 2; // 0 to 1
        const currentRadius = this.radius + pulse * 2;
        const opacity = 0.5 + pulse * 0.5;

        ctx.beginPath();
        ctx.arc(this.x, this.y, currentRadius, 0, Math.PI * 2);
        ctx.fillStyle = this.color;
        ctx.globalAlpha = opacity;
        ctx.fill();
        
        // Glow effect
        ctx.shadowBlur = 10;
        ctx.shadowColor = this.color;
        ctx.fill();
        ctx.shadowBlur = 0;
        ctx.globalAlpha = 1;
      }
    }

    // Initialization
    const init = () => {
      width = container.clientWidth;
      height = container.clientHeight;
      canvas.width = width;
      canvas.height = height;
      
      nodes.length = 0;
      for (let i = 0; i < config.nodeCount; i++) {
        nodes.push(new Node());
      }
    };

    // Animation Loop
    const animate = () => {
      ctx.clearRect(0, 0, width, height);
      
      // Update and draw connections first (behind nodes)
      ctx.lineWidth = 1;
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const dx = nodes[i].x - nodes[j].x;
          const dy = nodes[i].y - nodes[j].y;
          const distance = Math.sqrt(dx * dx + dy * dy);

          if (distance < config.connectionDistance) {
            const opacity = 1 - (distance / config.connectionDistance);
            
            // Create gradient for connection
            const gradient = ctx.createLinearGradient(nodes[i].x, nodes[i].y, nodes[j].x, nodes[j].y);
            gradient.addColorStop(0, `${nodes[i].color}00`);
            gradient.addColorStop(0.5, `rgba(251, 191, 36, ${opacity * 0.5})`); // Gold center
            gradient.addColorStop(1, `${nodes[j].color}00`);

            ctx.beginPath();
            ctx.strokeStyle = gradient;
            ctx.moveTo(nodes[i].x, nodes[i].y);
            ctx.lineTo(nodes[j].x, nodes[j].y);
            ctx.stroke();

            // Data particles flowing along connections
            if (opacity > 0.6) {
              const time = Date.now() * 0.001;
              const particlePos = (time % 1); // 0 to 1
              const px = nodes[i].x + (nodes[j].x - nodes[i].x) * particlePos;
              const py = nodes[i].y + (nodes[j].y - nodes[i].y) * particlePos;
              
              ctx.beginPath();
              ctx.fillStyle = '#dc2626'; // Red data packet
              ctx.arc(px, py, 1.5, 0, Math.PI * 2);
              ctx.fill();
            }
          }
        }
      }

      // Draw nodes
      nodes.forEach(node => {
        node.update();
        node.draw();
      });

      animationFrameId = requestAnimationFrame(animate);
    };

    // Event Listeners
    const handleResize = () => init();
    const handleMouseMove = (e) => {
      const rect = canvas.getBoundingClientRect();
      mouseX = e.clientX - rect.left;
      mouseY = e.clientY - rect.top;
    };
    const handleMouseLeave = () => {
      mouseX = -1000;
      mouseY = -1000;
    };

    window.addEventListener('resize', handleResize);
    canvas.addEventListener('mousemove', handleMouseMove);
    canvas.addEventListener('mouseleave', handleMouseLeave);

    init();
    animate();

    return () => {
      window.removeEventListener('resize', handleResize);
      canvas.removeEventListener('mousemove', handleMouseMove);
      canvas.removeEventListener('mouseleave', handleMouseLeave);
      cancelAnimationFrame(animationFrameId);
    };
  }, []);

  return (
    <motion.div 
      ref={containerRef}
      className="relative w-full h-[400px] md:h-[500px] rounded-xl overflow-hidden border border-white/10 bg-navy-950/50 backdrop-blur-sm shadow-2xl"
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 1 }}
    >
      {/* Background Grid */}
      <div className="absolute inset-0 z-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:24px_24px]" />
      
      {/* Vignette */}
      <div className="absolute inset-0 bg-radial-gradient from-transparent via-navy-950/20 to-navy-950/80 z-10 pointer-events-none" />

      {/* Main Visualization Canvas */}
      <canvas 
        ref={canvasRef}
        className="absolute inset-0 z-20 w-full h-full"
      />

      {/* Overlay UI Elements - to make it look like a dashboard */}
      <div className="absolute top-4 left-4 z-30 flex items-center gap-2">
        <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
        <span className="text-xs font-mono text-green-500/80">SYSTEM ONLINE</span>
      </div>
      
      <div className="absolute bottom-4 right-4 z-30 flex flex-col items-end gap-1">
        <span className="text-[10px] font-mono text-gold-500/60">NODES: ACTIVE</span>
        <span className="text-[10px] font-mono text-red-500/60">LATENCY: &lt;1ms</span>
      </div>

      {/* Central Holographic Label */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 z-30 pointer-events-none text-center">
        <div className="relative">
          <div className="absolute -inset-4 bg-navy-950/80 blur-xl rounded-full" />
          <h3 className="relative text-sm font-bold tracking-[0.2em] text-gold-400 opacity-80">
            NEURAL<br/>CORE
          </h3>
        </div>
      </div>
    </motion.div>
  );
};

export default HeroImage;