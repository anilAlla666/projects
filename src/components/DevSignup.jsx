import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Link } from 'react-router-dom';
import { ArrowLeft } from 'lucide-react';

const DevSignup = () => {
  const [isSubmitted, setIsSubmitted] = useState(false);
  const [formData, setFormData] = useState({
    name: '',
    email: ''
  });

  const handleSubmit = (e) => {
    e.preventDefault();
    // Simulate API call
    setTimeout(() => {
      setIsSubmitted(true);
      // In a real app, we would send formData to a backend here
      localStorage.setItem('dev_waitlist', JSON.stringify(formData));
    }, 500);
  };

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  return (
    <div className="min-h-screen bg-[#F9FAFB] flex flex-col items-center justify-center p-4">
      <div className="w-full max-w-md">
        <Link to="/" className="inline-flex items-center text-gray-500 hover:text-[#6366F1] mb-8 transition-colors">
          <ArrowLeft className="w-4 h-4 mr-2" />
          Back to Home
        </Link>
        
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="bg-white p-8 rounded-xl shadow-lg border border-gray-100"
        >
          {!isSubmitted ? (
            <>
              <h1 
                className="text-3xl font-bold text-[#111827] mb-2 text-center"
                style={{ fontFamily: "'Styrene', sans-serif" }}
              >
                Developer Access
              </h1>
              <p className="text-gray-500 text-center mb-8 font-inter">
                Join the HyperFlux kernel beta program.
              </p>

              <form onSubmit={handleSubmit} className="space-y-6">
                <div className="space-y-2">
                  <Label htmlFor="name">Full Name</Label>
                  <Input 
                    id="name" 
                    name="name" 
                    placeholder="Ada Lovelace" 
                    required 
                    value={formData.name}
                    onChange={handleChange}
                    className="font-inter"
                  />
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="email">Work Email</Label>
                  <Input 
                    id="email" 
                    name="email" 
                    type="email" 
                    placeholder="ada@institute.org" 
                    required 
                    value={formData.email}
                    onChange={handleChange}
                    className="font-inter"
                  />
                </div>

                <Button 
                  type="submit" 
                  className="w-full bg-[#6366F1] hover:bg-[#4F46E5] text-white py-6 text-lg font-inter transition-all duration-300"
                >
                  Join Waitlist
                </Button>
              </form>
            </>
          ) : (
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className="text-center py-8"
            >
              <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-6">
                <svg className="w-8 h-8 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              </div>
              <h2 
                className="text-2xl font-bold text-[#111827] mb-4"
                style={{ fontFamily: "'Styrene', sans-serif" }}
              >
                You're on the list!
              </h2>
              <p className="text-gray-600 mb-8 font-inter">
                Thank you for joining the waitlist, we will update you once ready.
              </p>
              <Button 
                asChild
                className="bg-gray-100 hover:bg-gray-200 text-gray-900 font-inter"
              >
                <Link to="/">Return to Homepage</Link>
              </Button>
            </motion.div>
          )}
        </motion.div>
        
        <p className="text-center text-gray-400 text-sm mt-8 font-inter">
          © {new Date().getFullYear()} HyperFlux Research Institute
        </p>
      </div>
    </div>
  );
};

export default DevSignup;