import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { useToast } from '@/hooks/use-toast';
import { supabase } from '@/lib/customSupabaseClient';

const ContactFormModal = ({ isOpen, onClose }) => {
  const [loading, setLoading] = useState(false);
  const { toast } = useToast();
  const [formData, setFormData] = useState({ name: '', email: '', company: '', message: '' });

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const { data, error } = await supabase.functions.invoke('send-contact-email', {
        body: formData
      });
      
      if (error) throw error;

      toast({
        title: "Message Sent",
        description: "We've received your request and will be in touch soon.",
      });
      onClose();
      setFormData({ name: '', email: '', company: '', message: '' });
    } catch (err) {
      toast({
        title: "Error",
        description: "Failed to send message. Please try again.",
        variant: "destructive"
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-background/80 backdrop-blur-sm z-[200]"
            onClick={onClose}
          />
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 20 }}
            className="fixed left-[50%] top-[50%] z-[201] w-[95%] sm:w-[90%] md:w-[500px] lg:w-[600px] translate-x-[-50%] translate-y-[-50%] p-2 sm:p-4 md:p-6"
          >
            <div className="bg-card border border-border rounded-xl shadow-2xl p-4 sm:p-6 md:p-8 relative max-h-[90vh] overflow-y-auto">
              <button 
                onClick={onClose} 
                className="absolute right-2 sm:right-4 top-2 sm:top-4 text-[#a1a1aa] hover:text-[#fafafa] transition-colors min-w-[44px] min-h-[44px] flex items-center justify-center"
                aria-label="Close modal"
              >
                <X className="w-5 h-5 sm:w-6 sm:h-6" />
              </button>
              
              <h2 className="text-2xl sm:text-3xl font-serif text-[#fafafa] mb-2 pr-10">Get Access</h2>
              <p className="text-sm sm:text-base text-[#a1a1aa] font-sans mb-6 sm:mb-8 pr-4">
                Reach out to <a href="mailto:anilkumaralla@neuraldynamicsteam.io" className="text-[#f59e0b] hover:underline">anilkumaralla@neuraldynamicsteam.io</a> or use the form below.
              </p>
              
              <form onSubmit={handleSubmit} className="space-y-4 sm:space-y-5">
                <div className="space-y-1.5 sm:space-y-2">
                  <Label htmlFor="name" className="text-sm sm:text-base text-[#fafafa]">Name</Label>
                  <Input id="name" required className="bg-background text-[#fafafa] min-h-[44px] text-base" value={formData.name} onChange={e => setFormData({...formData, name: e.target.value})} />
                </div>
                <div className="space-y-1.5 sm:space-y-2">
                  <Label htmlFor="email" className="text-sm sm:text-base text-[#fafafa]">Work Email</Label>
                  <Input id="email" type="email" required className="bg-background text-[#fafafa] min-h-[44px] text-base" value={formData.email} onChange={e => setFormData({...formData, email: e.target.value})} />
                </div>
                <div className="space-y-1.5 sm:space-y-2">
                  <Label htmlFor="company" className="text-sm sm:text-base text-[#fafafa]">Company</Label>
                  <Input id="company" className="bg-background text-[#fafafa] min-h-[44px] text-base" value={formData.company} onChange={e => setFormData({...formData, company: e.target.value})} />
                </div>
                <div className="space-y-1.5 sm:space-y-2">
                  <Label htmlFor="message" className="text-sm sm:text-base text-[#fafafa]">Message</Label>
                  <Textarea id="message" required className="bg-background text-[#fafafa] min-h-[100px] text-base resize-y" value={formData.message} onChange={e => setFormData({...formData, message: e.target.value})} />
                </div>
                <Button type="submit" disabled={loading} className="w-full bg-[#f59e0b] text-black hover:bg-[#fbbf24] font-bold min-h-[44px] text-base sm:text-lg mt-2">
                  {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : "Submit Request"}
                </Button>
              </form>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
};

export default ContactFormModal;