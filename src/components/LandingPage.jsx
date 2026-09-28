
import React from 'react';
import Hero from '@/components/Hero';
import ProductsSection from '@/components/ProductsSection';
import AboutSection from '@/components/AboutSection';
import ContactSection from '@/components/ContactSection';
import Footer from '@/components/Footer';

const LandingPage = () => {
  return (
    <div className="w-full flex flex-col min-h-screen">
      <Hero id="hero" />
      <ProductsSection id="products" />
      <AboutSection id="about" />
      <ContactSection id="contact" />
      <Footer id="footer" />
    </div>
  );
};

export default LandingPage;
