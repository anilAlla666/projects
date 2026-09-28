import { cn } from '@/lib/utils';
import { Slot } from '@radix-ui/react-slot';
import { cva } from 'class-variance-authority';
import React from 'react';
import { motion } from 'framer-motion';

const buttonVariants = cva(
	'relative inline-flex items-center justify-center rounded-lg text-sm font-medium ring-offset-background transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 overflow-hidden',
	{
		variants: {
			variant: {
				default: 'bg-[#f59e0b] hover:bg-[#d97706] text-black shadow-lg hover:shadow-[#f59e0b]/40 border border-[#f59e0b]',
				destructive:
          'bg-destructive text-destructive-foreground hover:bg-destructive/90',
				outline:
          'border border-[#f59e0b]/30 bg-transparent hover:bg-[#f59e0b]/10 text-[#f59e0b] hover:text-[#fbbf24] backdrop-blur-sm',
				secondary:
          'bg-secondary text-secondary-foreground hover:bg-secondary/80',
				ghost: 'hover:bg-accent hover:text-accent-foreground',
				link: 'text-primary underline-offset-4 hover:underline',
        premium: 'bg-gradient-to-r from-[#f59e0b] via-[#fbbf24] to-[#d97706] text-black font-bold shadow-lg hover:shadow-xl border border-[#fbbf24]/50'
			},
			size: {
				default: 'h-11 px-6 py-2',
				sm: 'h-9 rounded-md px-3',
				lg: 'h-14 rounded-lg px-10 text-lg',
				icon: 'h-10 w-10',
			},
		},
		defaultVariants: {
			variant: 'default',
			size: 'default',
		},
	},
);

const Button = React.forwardRef(({ className, variant, size, asChild = false, children, ...props }, ref) => {
	const Comp = asChild ? Slot : motion.button;
  
  const motionProps = asChild ? {} : {
    whileHover: { scale: 1.02, y: -2 },
    whileTap: { scale: 0.98, y: 0 },
    transition: { type: "spring", stiffness: 400, damping: 17 }
  };

	return (
		<Comp
			className={cn(buttonVariants({ variant, size, className }))}
			ref={ref}
			{...motionProps}
      {...props}
		>
      {/* Chrome shine effect for premium buttons */}
      {variant === 'premium' && (
        <span className="absolute inset-0 w-full h-full bg-gradient-to-r from-transparent via-white/40 to-transparent -translate-x-full group-hover:animate-[shimmer_2s_infinite] skew-x-12" />
      )}
      <span className="relative flex items-center gap-2 z-10">
        {children}
      </span>
		</Comp>
	);
});
Button.displayName = 'Button';

export { Button, buttonVariants };