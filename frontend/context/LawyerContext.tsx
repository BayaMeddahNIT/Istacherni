/**
 * LawyerContext.tsx
 * -----------------
 * Global shared state for lawyers.
 *
 * The map screen WRITES lawyers into this context after fetching them.
 * The list screen READS from this context — no extra API call needed.
 */

import React, {
  createContext,
  useContext,
  useState,
  useMemo,
  ReactNode,
  useCallback,
} from "react";
import { Lawyer } from "@/services/lawyerService";

// ─── Types ────────────────────────────────────────────────────────────────────

export interface LawyerFilters {
  search: string;   // name or specialty
  city: string;     // city filter (empty = all)
}

interface LawyerContextType {
  /** Raw lawyers list (set by map screen). */
  lawyers: Lawyer[];
  /** True while the map is fetching lawyers. */
  loading: boolean;
  /** Whether lawyers came from Google Places (true) or mock data (false). */
  fromGoogle: boolean;
  /** Local search / filter state for the list screen. */
  filters: LawyerFilters;
  /** Lawyers after applying the current filters. */
  filteredLawyers: Lawyer[];
  /** Called by the map screen after it resolves lawyers. */
  setLawyers: (lawyers: Lawyer[], fromGoogle: boolean) => void;
  /** Called by the map screen while fetching. */
  setLoading: (loading: boolean) => void;
  /** Update local search / filter state. */
  updateFilters: (patch: Partial<LawyerFilters>) => void;
  /** User's current GPS coordinates (set by map screen). */
  userCoords: { latitude: number; longitude: number } | null;
  setUserCoords: (coords: { latitude: number; longitude: number }) => void;
}

// ─── Context ──────────────────────────────────────────────────────────────────

const LawyerContext = createContext<LawyerContextType | undefined>(undefined);

// ─── Provider ─────────────────────────────────────────────────────────────────

export function LawyerProvider({ children }: { children: ReactNode }) {
  // Start empty — populated by the map screen after location resolves.
  const [lawyers, setLawyersState] = useState<Lawyer[]>([]);
  const [loading, setLoadingState] = useState(false);
  const [fromGoogle, setFromGoogle] = useState(false);
  const [filters, setFilters] = useState<LawyerFilters>({ search: "", city: "" });
  const [userCoords, setUserCoordsState] = useState<{ latitude: number; longitude: number } | null>(null);

  const setLawyers = useCallback((newLawyers: Lawyer[], google: boolean) => {
    setLawyersState(newLawyers);
    setFromGoogle(google);
  }, []);

  const setLoading = useCallback((val: boolean) => {
    setLoadingState(val);
  }, []);

  const updateFilters = useCallback((patch: Partial<LawyerFilters>) => {
    setFilters((prev) => ({ ...prev, ...patch }));
  }, []);

  const setUserCoords = useCallback((coords: { latitude: number; longitude: number }) => {
    setUserCoordsState(coords);
  }, []);

  // Apply filters with memoization — re-computed only when lawyers or filters change.
  const filteredLawyers = useMemo(() => {
    const searchLower = filters.search.toLowerCase().trim();
    const cityLower = filters.city.toLowerCase().trim();
    return lawyers.filter((l) => {
      const matchesSearch =
        !searchLower ||
        l.name.toLowerCase().includes(searchLower) ||
        l.specialty.toLowerCase().includes(searchLower);
      const matchesCity =
        !cityLower || l.city.toLowerCase().includes(cityLower);
      return matchesSearch && matchesCity;
    });
  }, [lawyers, filters]);

  const value = useMemo<LawyerContextType>(
    () => ({
      lawyers,
      loading,
      fromGoogle,
      filters,
      filteredLawyers,
      setLawyers,
      setLoading,
      updateFilters,
      userCoords,
      setUserCoords,
    }),
    [lawyers, loading, fromGoogle, filters, filteredLawyers, setLawyers, setLoading, updateFilters, userCoords, setUserCoords]
  );

  return (
    <LawyerContext.Provider value={value}>{children}</LawyerContext.Provider>
  );
}

// ─── Raw Hook ─────────────────────────────────────────────────────────────────

export function useLawyerContext(): LawyerContextType {
  const ctx = useContext(LawyerContext);
  if (!ctx) throw new Error("useLawyerContext must be used within LawyerProvider");
  return ctx;
}
