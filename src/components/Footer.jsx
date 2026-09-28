
import React from 'react';

const Footer = ({ id }) => {
  const scrollToSection = (sectionId) => {
    const element = document.getElementById(sectionId);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <footer id={id} className="bg-background border-t py-12">
      <div className="container mx-auto px-6 max-w-6xl">
        <div className="flex flex-col md:flex-row justify-between items-center gap-8">
          
          <div className="text-center md:text-left">
            <h3 className="text-xl font-bold text-foreground mb-2">Neural Dynamics Team</h3>
            <p className="text-sm text-muted-foreground">O(N) becomes O(1).</p>
          </div>

          <div className="flex gap-6 text-sm font-medium text-muted-foreground">
            <button onClick={() => window.scrollTo(0,0)} className="hover:text-foreground transition-colors">Home</button>
            <button onClick={() => scrollToSection('products')} className="hover:text-foreground transition-colors">Products</button>
            <button onClick={() => scrollToSection('about')} className="hover:text-foreground transition-colors">About</button>
            <button onClick={() => scrollToSection('contact')} className="hover:text-foreground transition-colors">Contact</button>
          </div>

          <div className="text-center md:text-right text-sm text-muted-foreground">
            <a href="mailto:anilkumaralla@neuraldynamicsteam.io" className="block hover:text-foreground mb-2">
              anilkumaralla@neuraldynamicsteam.io
            </a>
            <p>© {new Date().getFullYear()} Neural Dynamics Team. All rights reserved.</p>
          </div>

        </div>
      </div>
    </footer>
  );
};

export default Footer;
