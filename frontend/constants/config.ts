// ─── App Configuration ───────────────────────────────────────────────────────
//
// GOOGLE PLACES API KEY
// ─────────────────────
// To enable real lawyer data from Google Maps:
//
// 1. Go to: https://console.cloud.google.com/
// 2. Create a project → Enable "Places API"
// 3. Generate an API key → paste it below
// 4. Restrict the key to: Android + iOS Bundle IDs (for production)
//
// FREE TIER: 2,500 requests/day ($0 cost). After that: ~$0.032/request.
// Typical usage: 1 request per user per app open = very low cost.
//
// Leave empty ("") to use the built-in demo data (with disclaimer).

export const GOOGLE_PLACES_API_KEY = "AIzaSyCD9MEbuAMTXNZlEeVOzi3xG2kEkUt4fBQ";
// Search radius in meters around the user's location
export const LAWYER_SEARCH_RADIUS_M = 8000; // 8 km

// Max number of lawyers to display from Google
export const LAWYER_MAX_RESULTS = 20;
