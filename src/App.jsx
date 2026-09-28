
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import LandingPage from '@/components/LandingPage';
import ProductDetailPage from '@/components/ProductDetailPage';
import { Toaster } from '@/components/ui/toaster';

export default function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/product/:productId" element={<ProductDetailPage />} />
      </Routes>
      <Toaster />
    </Router>
  );
}
