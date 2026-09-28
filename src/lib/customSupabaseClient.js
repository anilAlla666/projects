import { createClient } from '@supabase/supabase-js';

const supabaseUrl = 'https://uzbzupgiwmdnfyrcoqbm.supabase.co';
const supabaseAnonKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InV6Ynp1cGdpd21kbmZ5cmNvcWJtIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjU4MTI5MDYsImV4cCI6MjA4MTM4ODkwNn0.RutHGXipFOoTqOb_lfr-MAqOLH0SL_pPphf7DcrPCYM';

const customSupabaseClient = createClient(supabaseUrl, supabaseAnonKey);

export default customSupabaseClient;

export { 
    customSupabaseClient,
    customSupabaseClient as supabase,
};
