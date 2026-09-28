import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { ArrowLeft, Calendar, User, Tag, Clock } from 'lucide-react';
import { supabase } from '@/lib/customSupabaseClient';
import { Button } from '@/components/ui/button';

const BlogPostDetail = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const [post, setPost] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchPost = async () => {
      if (!id) return;
      
      try {
        const { data, error } = await supabase
          .from('blog_posts')
          .select('*')
          .eq('id', id)
          .single();

        if (error) throw error;
        setPost(data);
      } catch (error) {
        console.error('Error fetching post:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchPost();
    // Scroll to top when component mounts
    window.scrollTo(0, 0);
  }, [id]);

  if (loading) {
    return (
      <div className="min-h-screen bg-navy-950 flex items-center justify-center">
        <div className="w-16 h-16 border-4 border-gold-500/30 border-t-gold-500 rounded-full animate-spin"></div>
      </div>
    );
  }

  if (!post) {
    return (
      <div className="min-h-screen bg-navy-950 flex flex-col items-center justify-center text-white font-sans">
        <h2 className="text-2xl font-serif font-normal mb-4">Post not found</h2>
        <Button variant="outline" onClick={() => navigate('/')} className="font-sans">Return Home</Button>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-navy-950 relative overflow-hidden pt-20 pb-20">
      {/* Background Ambience */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-gold-500/5 rounded-full blur-[100px]" />
        <div className="absolute bottom-0 left-0 w-[500px] h-[500px] bg-blue-500/5 rounded-full blur-[100px]" />
      </div>

      <div className="max-w-4xl mx-auto px-4 sm:px-6 relative z-10">
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5 }}
          className="mb-8"
        >
          <Button 
            variant="ghost" 
            onClick={() => navigate('/#blog')}
            className="text-gray-400 font-sans hover:text-gold-400 hover:bg-white/5 gap-2"
          >
            <ArrowLeft className="w-4 h-4" /> Back to Insights
          </Button>
        </motion.div>

        <motion.article
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          {/* Header Image */}
          <div className="relative w-full h-[400px] rounded-2xl overflow-hidden mb-12 border border-white/10 shadow-2xl">
            <img 
              src={post.image_url} 
              alt={post.title} 
              className="w-full h-full object-cover"
            />
            <div className="absolute inset-0 bg-gradient-to-t from-navy-950 via-navy-950/40 to-transparent" />
            
            <div className="absolute bottom-0 left-0 p-8 w-full font-sans">
              <div className="flex flex-wrap gap-4 items-center text-sm font-semibold text-gold-400 mb-4">
                <span className="bg-gold-500/10 border border-gold-500/20 px-3 py-1 rounded-full backdrop-blur-md">
                  {post.category}
                </span>
                <div className="flex items-center gap-2 text-gray-300">
                  <Calendar className="w-4 h-4" />
                  {new Date(post.created_at).toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' })}
                </div>
                <div className="flex items-center gap-2 text-gray-300">
                  <Clock className="w-4 h-4" />
                  5 min read
                </div>
              </div>
              
              <h1 className="text-4xl md:text-5xl font-serif font-normal text-white leading-tight">
                {post.title}
              </h1>
            </div>
          </div>

          {/* Content */}
          <div className="glass-card p-8 md:p-12 rounded-2xl relative overflow-hidden font-sans">
            <div className="absolute top-0 right-0 p-6 opacity-20">
              <Tag className="w-24 h-24 text-white rotate-12" />
            </div>

            <div className="flex items-center gap-4 mb-8 pb-8 border-b border-white/10">
              <div className="w-12 h-12 bg-white/10 rounded-full flex items-center justify-center border border-white/20">
                <User className="w-6 h-6 text-gold-400" />
              </div>
              <div>
                <div className="text-white font-semibold">{post.author}</div>
                <div className="text-sm text-gray-400">HyperFlux Research Team</div>
              </div>
            </div>

            <div className="prose prose-invert prose-lg max-w-none">
              <p className="text-xl text-gray-300 font-light leading-relaxed mb-8 border-l-4 border-gold-500 pl-6 italic">
                {post.excerpt}
              </p>
              
              <div className="text-gray-300 space-y-6 leading-relaxed whitespace-pre-line">
                {post.content}
              </div>
            </div>
          </div>
        </motion.article>
      </div>
    </div>
  );
};

export default BlogPostDetail;