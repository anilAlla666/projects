import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Send, Loader2, Sparkles, AlertCircle, CheckCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { useToast } from '@/components/ui/use-toast';
import { supabase } from '@/lib/customSupabaseClient';
import AnimatedAIBackground from '@/components/AnimatedAIBackground';

const RolesPage = () => {
  const { toast } = useToast();
  const [formData, setFormData] = useState({ name: '', email: '', message: '' });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [wordCount, setWordCount] = useState(0);

  const handleMessageChange = (e) => {
    const text = e.target.value;
    const words = text.trim().split(/\s+/).filter(Boolean);
    
    if (words.length <= 100) {
      setFormData({ ...formData, message: text });
      setWordCount(words.length);
    } else {
      // Allow deleting even if over limit
      if (text.length < formData.message.length) {
         setFormData({ ...formData, message: text });
         setWordCount(words.length);
      }
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!formData.name || !formData.email || !formData.message) {
      toast({ title: "Incomplete Form", description: "Please fill in all required fields.", variant: "destructive" });
      return;
    }

    setIsSubmitting(true);
    try {
      const { error } = await supabase.from('roles_submissions').insert([
        {
          name: formData.name,
          email: formData.email,
          message: formData.message,
          created_at: new Date().toISOString()
        }
      ]);

      if (error) throw error;

      toast({ 
        title: "Application Received", 
        description: "Your profile has been indexed. We will analyze your signal.",
        action: <CheckCircle className="text-green-500 w-5 h-5" />
      });
      setFormData({ name: '', email: '', message: '' });
      setWordCount(0);
    } catch (error) {
      console.error('Error submitting application:', error);
      toast({ 
        title: "Transmission Failed", 
        description: error.message || "Unable to send data. Please try again.", 
        variant: "destructive",
        action: <AlertCircle className="text-white w-5 h-5" />
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen relative overflow-hidden bg-navy-950 text-white pt-24 pb-20">
      <AnimatedAIBackground />
      
      <div className="container mx-auto px-4 relative z-10 max-w-4xl">
        {/* Header Section */}
        <motion.div 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-center mb-16"
        >
          <div className="inline-flex font-sans items-center justify-center p-2 px-4 rounded-full border border-gold-500/20 bg-gold-500/5 text-gold-400 text-sm font-medium mb-6 backdrop-blur-md">
            <Sparkles className="w-4 h-4 mr-2" />
            Career Opportunities
          </div>
          <h1 className="text-5xl md:text-7xl font-serif font-normal mb-6 tracking-tight">
            Build the <span className="text-transparent bg-clip-text bg-gradient-to-r from-gold-300 via-gold-500 to-red-500 animate-shimmer bg-[length:200%_auto] font-serif">Impossible</span>
          </h1>
          <p className="text-xl font-sans text-gray-400 max-w-2xl mx-auto leading-relaxed">
            We don't have "jobs". We have missions. Join a team dedicated to solving the hardest problems in distributed systems and AI infrastructure.
          </p>
        </motion.div>

        {/* Application Form Card */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="glass-card-gold p-8 md:p-12 rounded-3xl relative overflow-hidden font-sans"
        >
          <div className="absolute top-0 right-0 p-6 opacity-20 pointer-events-none">
             <div className="w-32 h-32 bg-gold-500/30 rounded-full blur-3xl"></div>
          </div>

          <div className="mb-8">
            <h2 className="text-2xl font-serif font-normal text-white mb-2">Initiate Contact</h2>
            <p className="text-gray-400 text-sm">Submit your credentials for review. Our kernel will process your application.</p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-8">
            <div className="grid md:grid-cols-2 gap-8">
              <div className="space-y-3">
                <Label htmlFor="name" className="text-gray-300 text-sm font-medium ml-1">Full Name</Label>
                <Input
                  id="name"
                  value={formData.name}
                  onChange={e => setFormData({...formData, name: e.target.value})}
                  placeholder="e.g. Ada Lovelace"
                  className="bg-navy-950/60 border-white/10 text-white h-12 transition-all duration-300 focus:border-gold-500/50 focus:ring-1 focus:ring-gold-500/20 placeholder:text-gray-600 font-sans"
                />
              </div>
              <div className="space-y-3">
                <Label htmlFor="email" className="text-gray-300 text-sm font-medium ml-1">Email Coordinates</Label>
                <Input
                  id="email"
                  value={formData.email}
                  onChange={e => setFormData({...formData, email: e.target.value})}
                  placeholder="signal@example.com"
                  type="email"
                  className="bg-navy-950/60 border-white/10 text-white h-12 transition-all duration-300 focus:border-gold-500/50 focus:ring-1 focus:ring-gold-500/20 placeholder:text-gray-600 font-sans"
                />
              </div>
            </div>

            <div className="space-y-3">
              <div className="flex justify-between items-center ml-1">
                <Label htmlFor="message" className="text-gray-300 text-sm font-medium">About You</Label>
                <span className={`text-xs ${wordCount >= 100 ? 'text-red-400' : 'text-gray-500'}`}>
                  {wordCount} / 100 words
                </span>
              </div>
              <Textarea
                id="message"
                value={formData.message}
                onChange={handleMessageChange}
                placeholder="Tell us what drives you. Why HyperFlux? (Max 100 words)"
                className="bg-navy-950/60 border-white/10 text-white min-h-[180px] transition-all duration-300 focus:border-gold-500/50 focus:ring-1 focus:ring-gold-500/20 placeholder:text-gray-600 resize-none leading-relaxed font-sans"
              />
            </div>

            <div className="pt-4 flex flex-col md:flex-row items-center gap-6">
              <Button 
                type="submit" 
                disabled={isSubmitting}
                className="w-full md:w-auto h-12 px-8 bg-gradient-to-r from-gold-500 to-red-600 hover:from-gold-400 hover:to-red-500 text-white font-medium font-sans rounded-lg shadow-lg hover:shadow-gold-500/20 transition-all duration-300 disabled:opacity-70 disabled:cursor-not-allowed"
              >
                {isSubmitting ? (
                  <span className="flex items-center">
                    <Loader2 className="animate-spin mr-2 h-4 w-4" /> Processing...
                  </span>
                ) : (
                  <span className="flex items-center">
                    Submit Application <Send className="ml-2 h-4 w-4" />
                  </span>
                )}
              </Button>
              
              <p className="text-xs text-gray-500 italic text-center md:text-left">
                * By submitting, you agree to our privacy protocols regarding data retention.
              </p>
            </div>
          </form>
        </motion.div>
      </div>
    </div>
  );
};

export default RolesPage;