import React, { useEffect, useRef } from 'react';
import { motion, useScroll, useTransform } from 'framer-motion';

const AnimatedAIBackground = () => {
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const { scrollY } = useScroll();
  
  // Parallax transform for the container itself
  // Moving from 0 to -200px over a long scroll distance
  const y = useTransform(scrollY, [0, 5000], [0, -200]); 
  
  // We track scroll position in a ref to use inside the requestAnimationFrame loop without re-triggering effects
  const scrollRef = useRef(0);
  
  useEffect(() => {
    return scrollY.onChange((latest) => {
      scrollRef.current = latest;
    });
  }, [scrollY]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d', { alpha: true });
    let width, height;
    let animationFrameId;
    let time = 0;

    // High saturation colors as requested
    const colors = ['#001a4d', '#FFD700', '#FF4444']; // Navy, Gold, Red
    
    // Layer Configuration
    // Reduced count, speed, and opacity by ~25% as requested
    const layerConfig = [
      { count: 26, speed: 0.11, scale: 0.5, opacity: 0.15, connectDist: 150, parallaxFactor: 0.05 },
      { count: 22, speed: 0.22, scale: 0.7, opacity: 0.22, connectDist: 180, parallaxFactor: 0.1 },
      { count: 18, speed: 0.37, scale: 0.9, opacity: 0.37, connectDist: 220, parallaxFactor: 0.2 },
      { count: 15, speed: 0.6, scale: 1.2, opacity: 0.52, connectDist: 260, parallaxFactor: 0.4 }
    ];

    const layerParticles = [[], [], [], []];
    const hexagons = [];

    class Particle {
      constructor(layerIndex) {
        this.layerIndex = layerIndex;
        this.reset(true);
      }

      reset(initial = false) {
        const layer = layerConfig[this.layerIndex];
        // Allow spawning outside visible area for smooth parallax entry
        this.x = Math.random() * width;
        this.y = Math.random() * height;
        
        // Velocity
        const angle = Math.random() * Math.PI * 2;
        this.vx = Math.cos(angle) * layer.speed;
        this.vy = Math.sin(angle) * layer.speed;
        
        // Appearance
        this.baseSize = (Math.random() * 3 + 2) * layer.scale * 1.5; // Base radius 3-7.5px -> Diameter 6-15px
        this.pulseOffset = Math.random() * Math.PI * 2;
        this.pulseSpeed = 0.015 + Math.random() * 0.02; // Slower pulse
        
        // Bias gold and red for upper layers
        if (this.layerIndex > 1 && Math.random() > 0.5) {
             this.color = Math.random() > 0.5 ? '#FFD700' : '#FF4444';
        } else {
             this.color = colors[Math.floor(Math.random() * colors.length)];
        }
      }

      update() {
        this.x += this.vx;
        this.y += this.vy;

        // Wrap around screen with buffer
        const buffer = 100;
        if (this.x < -buffer) this.x = width + buffer;
        if (this.x > width + buffer) this.x = -buffer;
        if (this.y < -buffer) this.y = height + buffer;
        if (this.y > height + buffer) this.y = -buffer;
      }

      draw(scrollOffset) {
        // Calculate parallax position
        const layer = layerConfig[this.layerIndex];
        const parallaxY = scrollOffset * layer.parallaxFactor;
        let drawY = this.y - parallaxY;

        // Infinite scroll wrapping for parallax vertical movement
        // If particle moves too far up due to scroll, wrap it to bottom
        const totalHeight = height + 200;
        while(drawY < -100) drawY += totalHeight;
        while(drawY > height + 100) drawY -= totalHeight;

        // Pulsing effect (Sine Wave)
        const pulse = Math.sin(time * this.pulseSpeed + this.pulseOffset);
        const currentSize = this.baseSize + pulse * (this.baseSize * 0.3);

        // Draw Glow Aura - Reduced intensity
        const gradient = ctx.createRadialGradient(this.x, drawY, currentSize * 0.5, this.x, drawY, currentSize * 2.5);
        gradient.addColorStop(0, this.color);
        gradient.addColorStop(1, 'rgba(0,0,0,0)');

        ctx.globalAlpha = layer.opacity * 0.75; // Further reduced global alpha for subtlety
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(this.x, drawY, currentSize * 2.5, 0, Math.PI * 2);
        ctx.fill();

        // Draw Core Node
        ctx.fillStyle = this.color;
        ctx.globalAlpha = Math.min(1, layer.opacity + 0.2); // Core is brighter but still reduced
        ctx.beginPath();
        ctx.arc(this.x, drawY, currentSize * 0.6, 0, Math.PI * 2);
        ctx.fill();
        
        return { x: this.x, y: drawY, color: this.color, opacity: layer.opacity, size: currentSize };
      }
    }

    class Hexagon {
      constructor() {
        this.reset();
      }
      reset() {
        this.x = Math.random() * width;
        this.y = Math.random() * height;
        this.size = Math.random() * 40 + 20; // 20-60px
        this.opacity = 0;
        this.maxOpacity = (0.05 + Math.random() * 0.1) * 0.75; // Reduced opacity
        this.fadeIn = true;
        this.life = Math.random() * 200 + 100;
        this.color = colors[Math.floor(Math.random() * colors.length)];
        this.rotation = Math.random() * Math.PI;
        this.rotationSpeed = (Math.random() - 0.5) * 0.003; // Slower rotation
      }
      update() {
        if (this.fadeIn) {
          this.opacity += 0.0015; // Slower fade in
          if (this.opacity >= this.maxOpacity) this.fadeIn = false;
        } else {
          this.opacity -= 0.0008; // Slower fade out
          if (this.opacity <= 0) this.reset();
        }
        this.rotation += this.rotationSpeed;
      }
      draw(scrollOffset) {
        // Hexagons stick to background layer (slow parallax)
        const drawY = this.y - scrollOffset * 0.02;
        
        // Wrap
        const totalHeight = height + 100;
        let finalY = drawY;
        while(finalY < -50) finalY += totalHeight;
        
        ctx.save();
        ctx.translate(this.x, finalY);
        ctx.rotate(this.rotation);
        
        ctx.strokeStyle = this.color;
        ctx.lineWidth = 1;
        ctx.globalAlpha = this.opacity;
        
        ctx.beginPath();
        for (let i = 0; i < 6; i++) {
          const angle = (Math.PI / 3) * i;
          const hx = this.size * Math.cos(angle);
          const hy = this.size * Math.sin(angle);
          if (i === 0) ctx.moveTo(hx, hy);
          else ctx.lineTo(hx, hy);
        }
        ctx.closePath();
        ctx.stroke();
        
        // Optional: Fill slightly
        ctx.fillStyle = this.color;
        ctx.globalAlpha = this.opacity * 0.15; // Reduced fill opacity
        ctx.fill();
        
        ctx.restore();
      }
    }

    const drawGrid = (scrollOffset) => {
      const gridSize = 80;
      const moveY = (scrollOffset * 0.05) % gridSize;
      
      ctx.strokeStyle = '#001a4d';
      ctx.lineWidth = 1;
      ctx.globalAlpha = 0.06; // Slightly reduced grid opacity
      
      // Vertical
      for (let x = 0; x < width; x += gridSize) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      }
      
      // Horizontal
      for (let y = -gridSize; y < height + gridSize; y += gridSize) {
        const dy = y - moveY;
        ctx.beginPath();
        ctx.moveTo(0, dy);
        ctx.lineTo(width, dy);
        ctx.stroke();
      }
    };

    const init = () => {
      width = window.innerWidth;
      height = window.innerHeight;
      
      const dpr = window.devicePixelRatio || 1;
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      ctx.scale(dpr, dpr);
      
      // Initialize Layer Particles
      layerConfig.forEach((cfg, i) => {
        layerParticles[i] = [];
        // Responsive count
        const count = width < 768 ? Math.floor(cfg.count * 0.6) : cfg.count;
        for (let j = 0; j < count; j++) {
          layerParticles[i].push(new Particle(i));
        }
      });
      
      // Initialize Hexagons
      hexagons.length = 0;
      for (let i = 0; i < 12; i++) { // Reduced count from 15 to 12
        hexagons.push(new Hexagon());
      }
    };

    const animate = () => {
      ctx.clearRect(0, 0, width, height);
      time++;
      const scrollOffset = scrollRef.current;
      
      // 1. Draw Geometric Background
      drawGrid(scrollOffset);
      hexagons.forEach(hex => {
        hex.update();
        hex.draw(scrollOffset);
      });
      
      // 2. Draw Network Layers
      layerParticles.forEach((particles, layerIndex) => {
        const layer = layerConfig[layerIndex];
        
        // Update & Draw Nodes
        const drawnNodes = particles.map(p => {
          p.update();
          return p.draw(scrollOffset);
        });
        
        // Draw Connections
        ctx.lineWidth = 1 + layerIndex * 0.3; // Thicker in foreground
        
        for (let i = 0; i < drawnNodes.length; i++) {
          for (let j = i + 1; j < drawnNodes.length; j++) {
            const p1 = drawnNodes[i];
            const p2 = drawnNodes[j];
            
            const dx = p1.x - p2.x;
            const dy = p1.y - p2.y;
            const dist = Math.sqrt(dx*dx + dy*dy);
            
            if (dist < layer.connectDist) {
              const opacity = (1 - dist / layer.connectDist) * layer.opacity * 0.75; // Reduced connection opacity
              
              if (opacity > 0.02) {
                // Gradient Connection
                const gradient = ctx.createLinearGradient(p1.x, p1.y, p2.x, p2.y);
                gradient.addColorStop(0, p1.color);
                gradient.addColorStop(1, p2.color);
                
                ctx.beginPath();
                ctx.strokeStyle = gradient;
                ctx.globalAlpha = opacity;
                ctx.moveTo(p1.x, p1.y);
                ctx.lineTo(p2.x, p2.y);
                ctx.stroke();
                
                // Flowing Data Packets
                // More packets for closer layers
                const packetCount = layerIndex + 1; // 1 to 4 packets
                
                for (let k = 0; k < packetCount; k++) {
                  // Different speeds and offsets - Reduced speed by 25%
                  const speed = (0.005 + (layerIndex * 0.002)) * 0.75; 
                  const offset = (time * speed + (i * j * 0.1) + (k / packetCount)) % 1;
                  
                  const px = p1.x + (p2.x - p1.x) * offset;
                  const py = p1.y + (p2.y - p1.y) * offset;
                  
                  // Bright packet
                  ctx.fillStyle = '#FFFFFF';
                  ctx.globalAlpha = (opacity + 0.2) * 0.8; // Reduced packet brightness
                  ctx.beginPath();
                  ctx.arc(px, py, 1.5 + layerIndex * 0.3, 0, Math.PI * 2);
                  ctx.fill();
                  
                  // Packet Glow
                  ctx.fillStyle = p1.color; // Use node color for glow
                  ctx.globalAlpha = opacity * 0.4; // Reduced glow
                  ctx.beginPath();
                  ctx.arc(px, py, 4 + layerIndex, 0, Math.PI * 2);
                  ctx.fill();
                }
              }
            }
          }
        }
      });
      
      ctx.globalAlpha = 1;
      animationFrameId = requestAnimationFrame(animate);
    };

    window.addEventListener('resize', init);
    init();
    animate();

    return () => {
      window.removeEventListener('resize', init);
      cancelAnimationFrame(animationFrameId);
    };
  }, []);

  return (
    <motion.div 
      ref={containerRef}
      className="fixed top-0 left-0 w-full h-full bg-navy-950 pointer-events-none"
      style={{ 
        zIndex: -10,
        y
      }}
    >
      {/* Deep Background Gradient */}
      <div className="absolute inset-0 bg-gradient-to-br from-[#020617] via-[#0B1120] to-[#0f172a] opacity-90" />
      
      {/* Canvas */}
      <canvas 
        ref={canvasRef} 
        className="absolute inset-0 w-full h-full mix-blend-screen"
        style={{ opacity: 0.8 }} 
      />
      
      {/* Contrast Overlay - Increased opacity to ensure text readability */}
      <div className="absolute inset-0 bg-navy-950/30 backdrop-blur-[1px]" />
    </motion.div>
  );
};

export default AnimatedAIBackground;