/**
 * supabase.js — Initialize Supabase client
 * ==========================================
 * REPLACE the two placeholder values below with your project's credentials.
 *
 * Where to find them:
 *   1. Log in at https://supabase.com
 *   2. Open your project → Settings → API
 *   3. Copy "Project URL"  → paste as SUPABASE_URL
 *   4. Copy "anon / public" key → paste as SUPABASE_ANON_KEY
 */

const SUPABASE_URL = "https://wtlrogzqemcftanjsgfp.supabase.co";   // e.g. https://xyzabc.supabase.co
const SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Ind0bHJvZ3pxZW1jZnRhbmpzZ2ZwIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzE2NzkwMTUsImV4cCI6MjA4NzI1NTAxNX0.aLs2hZFACO26IHDvDG6u5S-KWKo2cjuJooV0PAju1OY";      // long JWT string starting with eyJ…

const _supabase = supabase.createClient(SUPABASE_URL, SUPABASE_ANON_KEY);
