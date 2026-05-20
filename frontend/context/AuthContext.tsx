/**
 * AuthContext.tsx
 * ---------------
 * Lightweight auth state provider.
 *
 * - On mount, checks AsyncStorage for a saved access token.
 * - Exposes `isAuthenticated`, `isLoading`, and `logout()`.
 * - The root _layout.tsx reads `isAuthenticated` to gate navigation.
 * - Registers the `onAuthFailure` callback so expired tokens automatically
 *   kick the user back to the login screen.
 */

import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
} from "react";
import { router } from "expo-router";
import {
  clearTokens,
  getAccessToken,
  getSavedUsername,
  onAuthFailure,
  saveTokens,
} from "@/services/api";

// ── Types ──────────────────────────────────────────────────────────────────────

interface AuthState {
  isAuthenticated: boolean;
  isLoading: boolean;        // true while AsyncStorage is being checked
  username: string | null;
  login: (data: {
    access_token: string;
    refresh_token: string;
    user_id: number;
    username: string;
  }) => Promise<void>;
  logout: () => Promise<void>;
}

// ── Context ────────────────────────────────────────────────────────────────────

const AuthContext = createContext<AuthState | undefined>(undefined);

// ── Provider ───────────────────────────────────────────────────────────────────

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading]             = useState(true);  // start true
  const [username, setUsername]               = useState<string | null>(null);

  // ── On mount: read AsyncStorage once ─────────────────────────────────────
  useEffect(() => {
    (async () => {
      try {
        const token = await getAccessToken();
        if (token) {
          const name = await getSavedUsername();
          setIsAuthenticated(true);
          setUsername(name);
        }
      } catch {
        // Storage error → treat as logged out
      } finally {
        setIsLoading(false);
      }
    })();
  }, []);

  // ── Register the global 401 handler ──────────────────────────────────────
  useEffect(() => {
    onAuthFailure(async () => {
      setIsAuthenticated(false);
      setUsername(null);
      // Navigate outside the current render cycle
      setTimeout(() => router.replace("/(auth)/log-in"), 0);
    });
  }, []);

  // ── login / logout ────────────────────────────────────────────────────────
  const login = useCallback(async (data: {
    access_token: string;
    refresh_token: string;
    user_id: number;
    username: string;
  }) => {
    await saveTokens(data);
    setIsAuthenticated(true);
    setUsername(data.username);
  }, []);

  const logout = useCallback(async () => {
    await clearTokens();
    setIsAuthenticated(false);
    setUsername(null);
    router.replace("/(auth)/log-in");
  }, []);

  return (
    <AuthContext.Provider
      value={{ isAuthenticated, isLoading, username, login, logout }}
    >
      {children}
    </AuthContext.Provider>
  );
}

// ── Hook ───────────────────────────────────────────────────────────────────────

export function useAuth(): AuthState {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used inside <AuthProvider>");
  return ctx;
}
