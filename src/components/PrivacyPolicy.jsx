import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronDown, ChevronUp, Shield, Lock, Eye } from 'lucide-react';
import { cn } from "@/lib/utils";

const PrivacyPolicy = () => {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <section className="py-16 bg-[#0a0a0a] border-t border-gray-900" id="privacy">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-10">
          <div className="inline-flex items-center justify-center p-3 bg-indigo-500/10 rounded-full mb-4">
            <Shield className="w-6 h-6 text-indigo-400" />
          </div>
          <h2 className="text-3xl md:text-4xl font-bold text-[#F5F5DC] mb-4" style={{ fontFamily: "'Styrene', sans-serif" }}>
            Privacy Policy
          </h2>
          <p className="text-gray-400 max-w-2xl mx-auto text-sm">
            We are committed to protecting your personal data and ensuring transparency in how we handle your information.
          </p>
        </div>

        <div className="bg-[#111111] rounded-2xl border border-gray-800 overflow-hidden shadow-xl">
          <div className="p-6 md:p-8">
            <div className="prose prose-invert max-w-none prose-headings:text-[#F5F5DC] prose-p:text-gray-400 prose-li:text-gray-400 prose-strong:text-indigo-300">
              <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <Lock className="w-5 h-5 text-indigo-400" /> 
                1. Information Collection
              </h3>
              <p className="mb-4">
                We collect information you provide directly to us when you fill out our contact forms, sign up for our newsletter, or register for developer access. This includes your name, email address, and professional details relevant to your usage of HyperFlux technologies.
              </p>

              <div className={cn(
                "transition-all duration-500 overflow-hidden",
                isExpanded ? "max-h-[2000px] opacity-100" : "max-h-0 opacity-0"
              )}>
                <h3 className="text-xl font-semibold mt-6 mb-4 flex items-center gap-2">
                  <Eye className="w-5 h-5 text-indigo-400" />
                  2. Use of Information
                </h3>
                <p className="mb-4">
                  We use the information we collect to:
                </p>
                <ul className="list-disc pl-5 mb-4 space-y-2">
                  <li>Provide, maintain, and improve our services and kernel technologies.</li>
                  <li>Process your registration and verify your developer credentials.</li>
                  <li>Send you technical notices, updates, security alerts, and support messages.</li>
                  <li>Respond to your comments, questions, and customer service requests.</li>
                  <li>Communicate with you about products, services, offers, and events.</li>
                </ul>

                <h3 className="text-xl font-semibold mt-6 mb-4">3. Data Security</h3>
                <p className="mb-4">
                  We implement appropriate technical and organizational measures to protect the security of your personal data. However, please note that no system is completely secure, and we cannot guarantee the absolute security of your information during transmission or storage.
                </p>

                <h3 className="text-xl font-semibold mt-6 mb-4">4. Data Sharing</h3>
                <p className="mb-4">
                  We do not sell your personal data. We may share your information with third-party service providers who perform services on our behalf, such as cloud hosting (Supabase), analytics, and email delivery services. These providers are bound by confidentiality agreements and are restricted from using your data for any other purpose.
                </p>

                <h3 className="text-xl font-semibold mt-6 mb-4">5. Your Rights</h3>
                <p className="mb-4">
                  Depending on your location, you may have rights regarding your personal data, including the right to access, correct, delete, or restrict use of your data. To exercise these rights, please contact us through our provided support channels.
                </p>

                <h3 className="text-xl font-semibold mt-6 mb-4">6. Changes to this Policy</h3>
                <p className="mb-4">
                  We may update this privacy policy from time to time. If we make material changes, we will notify you by revising the date at the top of the policy and, in some cases, provide you with additional notice.
                </p>
                
                <p className="text-xs text-gray-500 mt-8 border-t border-gray-800 pt-4">
                  Last Updated: December 15, 2025
                </p>
              </div>
            </div>
          </div>

          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="w-full py-4 bg-gray-900/50 hover:bg-gray-900 border-t border-gray-800 flex items-center justify-center gap-2 text-sm font-medium text-indigo-400 transition-colors"
          >
            {isExpanded ? (
              <>
                Show Less <ChevronUp className="w-4 h-4" />
              </>
            ) : (
              <>
                Read Full Policy <ChevronDown className="w-4 h-4" />
              </>
            )}
          </button>
        </div>
      </div>
    </section>
  );
};

export default PrivacyPolicy;