import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Send, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { useToast } from '@/components/ui/use-toast';
import { supabase } from '@/lib/customSupabaseClient';

const ContactForm = ({ id }) => {
  const { toast } = useToast();
  const [formData, setFormData] = useState({ name: '', email: '', message: '' });
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsSubmitting(true);
    try {
      const { error } = await supabase.from('contact_submissions').insert([formData]);
      if (error) throw error;
      toast({ title: "Message Sent", description: "We'll be in touch shortly." });
      setFormData({ name: '', email: '', message: '' });
    } catch (error) {
      toast({ title: "Error", description: error.message, variant: "destructive" });
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <section className="py-32 relative perspective-1000" id={id}>
      <div className="max-w-4xl mx-auto px-4 relative z-10">
        <motion.div
          initial={{ rotateX: 20, opacity: 0 }}
          whileInView={{ rotateX: 0, opacity: 1 }}
          transition={{ duration: 0.8 }}
          className="glass-card-gold p-8 md:p-12 rounded-3xl transform-3d hover:shadow-luxury-lg transition-all duration-500"
        >
          <div className="text-center mb-10">
            <h2 className="text-4xl font-serif font-normal text-white mb-4">Initialize <span className="text-gold-400 font-serif">Contact</span></h2>
            <p className="text-gray-400 font-sans">Secure channel establishment for enterprise inquiries.</p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-8 font-sans">
            <div className="grid md:grid-cols-2 gap-8">
              <div className="space-y-2 group">
                <Label className="text-gray-300 font-medium ml-1">Identity</Label>
                <Input
                  value={formData.name}
                  onChange={e => setFormData({...formData, name: e.target.value})}
                  placeholder="Full Name"
                  className="bg-navy-950/50 border-white/10 text-white font-sans h-12 transition-all duration-300 focus:translate-z-[10px] focus:scale-[1.02] focus:border-gold-500/50"
                />
              </div>
              <div className="space-y-2 group">
                <Label className="text-gray-300 font-medium ml-1">Signal Trace</Label>
                <Input
                  value={formData.email}
                  onChange={e => setFormData({...formData, email: e.target.value})}
                  placeholder="Email Address"
                  type="email"
                  className="bg-navy-950/50 border-white/10 text-white font-sans h-12 transition-all duration-300 focus:translate-z-[10px] focus:scale-[1.02] focus:border-gold-500/50"
                />
              </div>
            </div>

            <div className="space-y-2 group">
              <Label className="text-gray-300 font-medium ml-1">Payload</Label>
              <Textarea
                value={formData.message}
                onChange={e => setFormData({...formData, message: e.target.value})}
                placeholder="Message content..."
                className="bg-navy-950/50 border-white/10 text-white font-sans min-h-[150px] transition-all duration-300 focus:translate-z-[10px] focus:scale-[1.02] focus:border-gold-500/50 resize-none"
              />
            </div>

            <div className="flex justify-end pt-4">
              <Button 
                type="submit" 
                disabled={isSubmitting}
                variant="premium"
                size="lg"
                className="w-full md:w-auto min-w-[200px] font-sans font-semibold hover:translate-z-[20px] transition-transform duration-300"
              >
                {isSubmitting ? <Loader2 className="animate-spin" /> : <>Transmit <Send className="ml-2 w-4 h-4" /></>}
              </Button>
            </div>
          </form>
        </motion.div>
      </div>
    </section>
  );
};

export default ContactForm;