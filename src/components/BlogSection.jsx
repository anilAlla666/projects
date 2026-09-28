import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Clock, ArrowRight } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { supabase } from '@/lib/customSupabaseClient';

const BlogCardSkeleton = () => (
  <div className="h-[400px] w-full bg-navy-900/40 border border-white/10 rounded-2xl overflow-hidden animate-pulse">
    <div className="h-1/2 bg-white/5" />
    <div className="p-6 space-y-4">
      <div className="h-4 bg-white/5 rounded w-1/3" />
      <div className="h-6 bg-white/5 rounded w-3/4" />
      <div className="h-4 bg-white/5 rounded w-full" />
      <div className="h-4 bg-white/5 rounded w-2/3" />
    </div>
  </div>
);

const BlogCard = ({ post, index }) => {
  const navigate = useNavigate();

  return (
    <div 
      className="h-[400px] w-full perspective-1000 group cursor-pointer" 
      onClick={() => navigate(`/blog/${post.id}`)}
    >
      <motion.div 
        className="relative w-full h-full duration-700 transform-3d group-hover:rotate-y-180"
        initial={{ opacity: 0, y: 50 }}
        whileInView={{ opacity: 1, y: 0 }}
        transition={{ delay: index * 0.1 }}
        viewport={{ once: true }}
      >
        {/* Front */}
        <div className="absolute inset-0 w-full h-full backface-hidden bg-navy-900/40 border border-white/10 rounded-2xl overflow-hidden shadow-lg hover:shadow-gold-500/20 transition-shadow">
          <div className="h-1/2 overflow-hidden">
            <img 
              src={post.image_url} 
              alt={post.title} 
              className="w-full h-full object-cover transition-transform duration-700 group-hover:scale-110"
            />
            <div className="absolute inset-0 bg-gradient-to-t from-navy-900 to-transparent" />
          </div>
          <div className="p-6">
            <div className="flex items-center font-sans gap-2 text-gold-400 text-xs font-bold uppercase tracking-wider mb-3">
              <span>{post.category}</span>
              <span>•</span>
              <span>{new Date(post.created_at).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}</span>
            </div>
            <h3 className="text-xl font-serif font-normal text-white mb-3 line-clamp-2">{post.title}</h3>
            <p className="text-gray-400 font-sans text-sm line-clamp-2">{post.excerpt}</p>
          </div>
        </div>

        {/* Back */}
        <div className="absolute inset-0 w-full h-full backface-hidden rotate-y-180 bg-navy-950 border border-gold-500/30 rounded-2xl p-8 flex flex-col justify-between shadow-luxury-lg">
          <div>
            <h3 className="text-xl font-serif font-normal text-gold-400 mb-4 line-clamp-2">{post.title}</h3>
            <p className="text-gray-300 font-sans text-sm leading-relaxed mb-6 line-clamp-4">{post.excerpt}</p>
            <div className="flex items-center font-sans gap-2 text-gray-500 text-xs">
              <Clock className="w-4 h-4" />
              <span>5 min read</span>
            </div>
          </div>
          <div className="flex items-center font-sans text-white font-bold text-sm group-hover:translate-x-2 transition-transform">
            Read Full Article <ArrowRight className="ml-2 w-4 h-4 text-gold-400" />
          </div>
        </div>
      </motion.div>
    </div>
  );
};

const BlogSection = ({ id }) => {
  const [posts, setPosts] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchPosts = async () => {
      try {
        const { data, error } = await supabase
          .from('blog_posts')
          .select('*')
          .order('created_at', { ascending: false });
        
        if (error) throw error;
        setPosts(data || []);
      } catch (error) {
        console.error('Error fetching blog posts:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchPosts();
  }, []);

  return (
    <section className="py-32 relative" id={id}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-serif font-normal text-white mb-6">
            Latest <span className="gradient-text font-serif">Insights</span>
          </h2>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {loading ? (
            // Show 3 skeletons while loading
            Array(3).fill(0).map((_, i) => <BlogCardSkeleton key={i} />)
          ) : (
            posts.map((post, index) => (
              <BlogCard key={post.id} post={post} index={index} />
            ))
          )}
        </div>
      </div>
    </section>
  );
};

export default BlogSection;