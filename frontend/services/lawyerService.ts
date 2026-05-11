import { GOOGLE_PLACES_API_KEY, LAWYER_SEARCH_RADIUS_M, LAWYER_MAX_RESULTS } from "@/constants/config";

// ─── Lawyer Interface ─────────────────────────────────────────────────────────

export interface Lawyer {
  id: string;
  name: string;
  specialty: string;
  phone: string;
  address: string;
  city: string;
  rating: number;
  ratingCount: number;
  experience: number;   // years — 0 = unknown from Google
  avatar: string;       // 2-letter initials
  available: boolean;
  source: "google" | "mock";
  mapsUrl?: string;
  photoRef?: string;
  email?: string;
  bio?: string;
  latitude?: number;
  longitude?: number;
}

// ─── Mock / Fallback Data ─────────────────────────────────────────────────────

export const MOCK_LAWYERS: Lawyer[] = [
  {
    id: "m1", name: "Maître Karim Boudiaf", specialty: "Droit du Travail",
    phone: "+213 555 123 456", address: "12 Rue Didouche Mourad", city: "Alger",
    rating: 4.8, ratingCount: 124, experience: 15, avatar: "KB",
    available: true, source: "mock",
    bio: "Spécialiste en droit du travail et conflits sociaux.",
    latitude: 36.7525, longitude: 3.04197,
  },
  {
    id: "m2", name: "Maître Samira Hadj", specialty: "Droit de la Famille",
    phone: "+213 555 234 567", address: "34 Avenue Krim Belkacem", city: "Alger",
    rating: 4.9, ratingCount: 87, experience: 12, avatar: "SH",
    available: true, source: "mock",
    bio: "Avocate dédiée au droit de la famille et au divorce.",
    latitude: 36.7645, longitude: 3.0565,
  },
  {
    id: "m3", name: "Maître Youcef Ziani", specialty: "Droit Commercial",
    phone: "+213 555 345 678", address: "7 Rue Ben M'hidi", city: "Oran",
    rating: 4.7, ratingCount: 203, experience: 20, avatar: "YZ",
    available: false, source: "mock",
    bio: "Expert en droit commercial et création d'entreprises.",
    latitude: 35.6969, longitude: -0.6331,
  },
  {
    id: "m4", name: "Maître Nadia Bensalem", specialty: "Droit Pénal",
    phone: "+213 555 456 789", address: "22 Rue de la Liberté", city: "Constantine",
    rating: 4.6, ratingCount: 56, experience: 8, avatar: "NB",
    available: true, source: "mock",
    bio: "Avocate pénaliste spécialisée en défense criminelle.",
    latitude: 36.365, longitude: 6.6147,
  },
  {
    id: "m5", name: "Maître Riad Meziane", specialty: "Droit Immobilier",
    phone: "+213 555 567 890", address: "5 Place du 1er Novembre", city: "Alger",
    rating: 4.5, ratingCount: 91, experience: 10, avatar: "RM",
    available: true, source: "mock",
    bio: "Conseil juridique en transactions immobilières.",
    latitude: 36.7725, longitude: 3.0455,
  },
];

// ─── In-Memory Cache ──────────────────────────────────────────────────────────

const _cache = new Map<string, { data: Lawyer[]; ts: number }>();
const CACHE_TTL_MS = 5 * 60 * 1000; // 5 minutes

function cacheKey(lat: number, lng: number) {
  return `${lat.toFixed(3)},${lng.toFixed(3)}`;
}

// ─── Places → Lawyer Mapper ────────────────────────────────────────────────────

function mapPlace(place: any): Lawyer {
  const name: string = place.name ?? "Cabinet d'avocat";
  const words = name.replace(/maître|cabinet|me\.|étude|société/gi, "").trim().split(/\s+/);
  const avatar = words.slice(0, 2).map((w: string) => w[0]?.toUpperCase() ?? "").join("") || "AV";
  const vicinity: string = place.vicinity ?? "";
  const parts = vicinity.split(",");
  const city = parts.length > 1 ? parts[parts.length - 1].trim() : "Algérie";
  const placeId: string = place.place_id ?? String(Math.random());
  const lat: number = place.geometry?.location?.lat ?? 0;
  const lng: number = place.geometry?.location?.lng ?? 0;

  return {
    id: placeId,
    name,
    specialty: "Avocat agréé",
    phone: "",
    address: vicinity,
    city,
    rating: place.rating ?? 0,
    ratingCount: place.user_ratings_total ?? 0,
    experience: 0,
    avatar,
    available: place.opening_hours?.open_now ?? true,
    source: "google",
    mapsUrl: `https://www.google.com/maps/search/?api=1&query=place_id:${placeId}`,
    photoRef: place.photos?.[0]?.photo_reference,
    latitude: lat,
    longitude: lng,
  };
}

// ─── Nearby Search (single keyword) ──────────────────────────────────────────

async function searchByKeyword(
  lat: number,
  lng: number,
  keyword: string,
): Promise<any[]> {
  const url =
    `https://maps.googleapis.com/maps/api/place/nearbysearch/json` +
    `?location=${lat},${lng}` +
    `&radius=${LAWYER_SEARCH_RADIUS_M}` +
    `&keyword=${encodeURIComponent(keyword)}` +
    `&language=fr` +
    `&key=${GOOGLE_PLACES_API_KEY}`;

  const res = await fetch(url);
  const data = await res.json();

  if (data.status === "REQUEST_DENIED") {
    console.warn(`[LawyerService] REQUEST_DENIED for keyword="${keyword}". Enable Places API on your key.`);
    throw new Error("REQUEST_DENIED");
  }
  if (data.status === "OK" && data.results?.length) return data.results as any[];
  return [];
}

// ─── Main Fetch Function ──────────────────────────────────────────────────────

export interface FetchLawyersResult {
  lawyers: Lawyer[];
  fromGoogle: boolean;
  error?: string;
}

/**
 * Fetches real nearby lawyers via Google Places.
 * Searches with 3 keywords ("avocat", "lawyer", "attorney") and deduplicates.
 * Falls back to MOCK_LAWYERS if the key is missing or denied.
 * Caches results for 5 minutes.
 */
export async function fetchNearbyLawyers(
  latitude: number,
  longitude: number,
): Promise<FetchLawyersResult> {
  const key = process.env.EXPO_PUBLIC_GOOGLE_PLACES_KEY || (GOOGLE_PLACES_API_KEY as string);
  if (!key || key.trim() === "") {
    return { lawyers: MOCK_LAWYERS, fromGoogle: false };
  }

  // Check cache
  const ck = cacheKey(latitude, longitude);
  const cached = _cache.get(ck);
  if (cached && Date.now() - cached.ts < CACHE_TTL_MS) {
    console.log("[LawyerService] Returning cached results.");
    return { lawyers: cached.data, fromGoogle: true };
  }

  const keywords = ["avocat", "lawyer", "attorney"];
  const seen = new Set<string>();
  const allPlaces: any[] = [];

  try {
    for (const kw of keywords) {
      const results = await searchByKeyword(latitude, longitude, kw);
      for (const r of results) {
        if (r.place_id && !seen.has(r.place_id)) {
          seen.add(r.place_id);
          allPlaces.push(r);
        }
      }
    }
  } catch (err: any) {
    if (err?.message === "REQUEST_DENIED") {
      return { lawyers: MOCK_LAWYERS, fromGoogle: false, error: "invalid_key" };
    }
    console.warn("[LawyerService] Fetch failed:", err);
    return { lawyers: MOCK_LAWYERS, fromGoogle: false, error: "network_error" };
  }

  if (!allPlaces.length) {
    return { lawyers: MOCK_LAWYERS, fromGoogle: false };
  }

  // Sort by rating descending, take top N
  allPlaces.sort((a, b) => (b.rating ?? 0) - (a.rating ?? 0));
  const lawyers: Lawyer[] = allPlaces
    .slice(0, LAWYER_MAX_RESULTS)
    .map(mapPlace);

  _cache.set(ck, { data: lawyers, ts: Date.now() });
  return { lawyers, fromGoogle: true };
}

// ─── Places Details: fetch phone + website ────────────────────────────────────

export interface LawyerDetails {
  phone: string | null;
  website: string | null;
}

const _detailsCache = new Map<string, LawyerDetails>();

export async function fetchLawyerDetails(placeId: string): Promise<LawyerDetails> {
  if (_detailsCache.has(placeId)) return _detailsCache.get(placeId)!;
  const key = process.env.EXPO_PUBLIC_GOOGLE_PLACES_KEY || (GOOGLE_PLACES_API_KEY as string);
  if (!key) return { phone: null, website: null };
  try {
    const url =
      `https://maps.googleapis.com/maps/api/place/details/json` +
      `?place_id=${placeId}` +
      `&fields=formatted_phone_number,website` +
      `&key=${key}`;
    const res = await fetch(url);
    const data = await res.json();
    const details: LawyerDetails = {
      phone: data.result?.formatted_phone_number ?? null,
      website: data.result?.website ?? null,
    };
    _detailsCache.set(placeId, details);
    return details;
  } catch {
    return { phone: null, website: null };
  }
}

/** Legacy compat — returns phone only */
export async function fetchLawyerPhone(placeId: string): Promise<string | null> {
  const d = await fetchLawyerDetails(placeId);
  return d.phone;
}

// ─── Directions API: real distance + ETA + polyline ──────────────────────────

export interface DirectionsResult {
  distanceText: string;   // "2.3 km"
  durationText: string;   // "6 min"
  distanceM: number;      // raw metres
  durationS: number;      // raw seconds
  polyline: { latitude: number; longitude: number }[]; // decoded route
}

/** Decodes a Google Maps encoded polyline string into lat/lng array. */
function decodePolyline(encoded: string): { latitude: number; longitude: number }[] {
  const coords: { latitude: number; longitude: number }[] = [];
  let index = 0;
  let lat = 0;
  let lng = 0;
  while (index < encoded.length) {
    let b: number;
    let shift = 0;
    let result = 0;
    do {
      b = encoded.charCodeAt(index++) - 63;
      result |= (b & 0x1f) << shift;
      shift += 5;
    } while (b >= 0x20);
    lat += (result & 1) !== 0 ? ~(result >> 1) : result >> 1;
    shift = 0; result = 0;
    do {
      b = encoded.charCodeAt(index++) - 63;
      result |= (b & 0x1f) << shift;
      shift += 5;
    } while (b >= 0x20);
    lng += (result & 1) !== 0 ? ~(result >> 1) : result >> 1;
    coords.push({ latitude: lat / 1e5, longitude: lng / 1e5 });
  }
  return coords;
}

export async function fetchDirections(
  originLat: number,
  originLng: number,
  destLat: number,
  destLng: number,
): Promise<DirectionsResult | null> {
  const key = process.env.EXPO_PUBLIC_GOOGLE_PLACES_KEY || (GOOGLE_PLACES_API_KEY as string);
  if (!key) return null;
  try {
    const url =
      `https://maps.googleapis.com/maps/api/directions/json` +
      `?origin=${originLat},${originLng}` +
      `&destination=${destLat},${destLng}` +
      `&mode=driving` +
      `&language=fr` +
      `&key=${key}`;
    const res = await fetch(url);
    const data = await res.json();
    if (data.status !== "OK" || !data.routes?.length) return null;
    const leg = data.routes[0].legs[0];
    const polyline = decodePolyline(data.routes[0].overview_polyline.points);
    return {
      distanceText: leg.distance.text,
      durationText: leg.duration.text,
      distanceM: leg.distance.value,
      durationS: leg.duration.value,
      polyline,
    };
  } catch (err) {
    console.warn("[LawyerService] Directions fetch failed:", err);
    return null;
  }
}
