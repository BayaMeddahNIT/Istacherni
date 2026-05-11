/**
 * api.ts
 * -------
 * Typed API client for the Istacherni backend.
 *
 * All requests automatically attach the stored JWT access token.
 * On 401 responses the client attempts a silent token refresh once,
 * then clears credentials and redirects to login on failure.
 */

import AsyncStorage from "@react-native-async-storage/async-storage";

// ── Config ─────────────────────────────────────────────────────────────────
export const API_BASE =
  process.env.EXPO_PUBLIC_API_URL || "http://172.32.31.30:8000";

const TIMEOUT_MS = 300_000; // 5 min (covers slow CPU inference)

// ── Storage keys ─────────────────────────────────────────────────────────────
const KEY_ACCESS = "@jwt_access";
const KEY_REFRESH = "@jwt_refresh";
const KEY_USER_ID = "@user_id";
const KEY_USERNAME = "@username";

// ── Token helpers ─────────────────────────────────────────────────────────────
export async function getAccessToken(): Promise<string | null> {
  return AsyncStorage.getItem(KEY_ACCESS);
}

export async function saveTokens(data: {
  access_token: string;
  refresh_token: string;
  user_id: number;
  username: string;
}) {
  await AsyncStorage.multiSet([
    [KEY_ACCESS, data.access_token],
    [KEY_REFRESH, data.refresh_token],
    [KEY_USER_ID, String(data.user_id)],
    [KEY_USERNAME, data.username],
  ]);
}

export async function clearTokens() {
  await AsyncStorage.multiRemove([KEY_ACCESS, KEY_REFRESH, KEY_USER_ID, KEY_USERNAME]);
}

export async function getSavedUsername(): Promise<string | null> {
  return AsyncStorage.getItem(KEY_USERNAME);
}

// ── Core fetch wrapper ────────────────────────────────────────────────────────
let _isRefreshing = false;
let _onRefreshFail: (() => void) | null = null;

/** Register a callback to be called when token refresh fails (e.g. navigate to login) */
export function onAuthFailure(cb: () => void) {
  _onRefreshFail = cb;
}

async function _tryRefresh(): Promise<string | null> {
  const refresh = await AsyncStorage.getItem(KEY_REFRESH);
  if (!refresh) return null;
  try {
    const res = await fetch(`${API_BASE}/auth/refresh`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ refresh_token: refresh }),
    });
    if (!res.ok) return null;
    const data = await res.json();
    await saveTokens(data);
    return data.access_token;
  } catch {
    return null;
  }
}

export async function apiFetch(
  path: string,
  options: RequestInit = {},
  retried = false
): Promise<Response> {
  const token = await getAccessToken();

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(options.headers as Record<string, string>),
  };
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const controller = new AbortController();
  const tid = setTimeout(() => controller.abort(), TIMEOUT_MS);

  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      ...options,
      headers,
      signal: controller.signal,
    });
  } finally {
    clearTimeout(tid);
  }

  // Auto-refresh on 401
  if (res.status === 401 && !retried && !_isRefreshing) {
    _isRefreshing = true;
    const newToken = await _tryRefresh();
    _isRefreshing = false;
    if (newToken) {
      return apiFetch(path, options, true);
    } else {
      await clearTokens();
      _onRefreshFail?.();
      return res;
    }
  }

  return res;
}

// ── Auth API ──────────────────────────────────────────────────────────────────
export async function apiRegister(username: string, email: string, password: string) {
  const res = await apiFetch("/auth/register", {
    method: "POST",
    body: JSON.stringify({ username, email, password }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Registration failed.");
  return data;
}

export async function apiLogin(username: string, password: string) {
  const res = await apiFetch("/auth/login", {
    method: "POST",
    body: JSON.stringify({ username, password }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Login failed.");
  await saveTokens(data);
  return data;
}

export async function apiLogout() {
  // Stateless JWT — just clear local storage
  await clearTokens();
}

export async function apiChangePassword(currentPassword: string, newPassword: string) {
  const res = await apiFetch("/auth/change-password", {
    method: "POST",
    body: JSON.stringify({ current_password: currentPassword, new_password: newPassword }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Password change failed.");
  return data;
}

export async function apiDeleteAccount() {
  const res = await apiFetch("/auth/account", { method: "DELETE" });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Account deletion failed.");
  await clearTokens();
  return data;
}

export async function apiFetchStats() {
  const res = await apiFetch("/auth/stats");
  if (!res.ok) throw new Error("Failed to load statistics.");
  return res.json();
}

// ── Session API ───────────────────────────────────────────────────────────────
export async function apiListSessions() {
  const res = await apiFetch("/sessions");
  if (!res.ok) throw new Error("Failed to load sessions.");
  return res.json();
}

export async function apiCreateSession() {
  const res = await apiFetch("/sessions", { method: "POST" });
  if (!res.ok) throw new Error("Failed to create session.");
  return res.json(); // { session_id, title }
}

export async function apiGetSession(sessionId: string) {
  const res = await apiFetch(`/sessions/${sessionId}`);
  if (!res.ok) throw new Error("Failed to load session.");
  return res.json();
}

export async function apiRenameSession(sessionId: string, title: string) {
  const res = await apiFetch(`/sessions/${sessionId}`, {
    method: "PATCH",
    body: JSON.stringify({ title }),
  });
  if (!res.ok) throw new Error("Failed to rename session.");
  return res.json();
}

export async function apiDeleteSession(sessionId: string) {
  const res = await apiFetch(`/sessions/${sessionId}`, { method: "DELETE" });
  if (!res.ok) throw new Error("Failed to delete session.");
  return res.json();
}

export async function apiDeleteAllSessions() {
  const res = await apiFetch("/sessions", { method: "DELETE" });
  if (!res.ok) throw new Error("Failed to delete all sessions.");
  return res.json();
}

// ── Chat API ──────────────────────────────────────────────────────────────────
export interface ChatPayload {
  question: string;
  top_k?: number;
  rag_type?: string;
  history?: { role: string; content: string }[];
  session_id?: string | null;
}

export async function apiChat(payload: ChatPayload) {
  const res = await apiFetch("/chat", {
    method: "POST",
    body: JSON.stringify({ message: payload.question, ...payload }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Chat request failed.");
  return data;
}

export async function apiSearch(question: string, top_k = 7, rag_type = "graph_local") {
  const res = await apiFetch("/search", {
    method: "POST",
    body: JSON.stringify({ question, top_k, rag_type }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Search failed.");
  return data; // { sources: [...] }
}

export async function apiContact(name: string, email: string, subject: string, message: string) {
  const res = await apiFetch("/contact", {
    method: "POST",
    body: JSON.stringify({ name, email, subject: subject.trim() || null, message }),
  });
  const data = await res.json();
  if (!res.ok) {
    if (res.status === 429) throw new Error("Too many requests. Please try again later.");
    throw new Error(data.detail || "Failed to send message.");
  }
  return data;
}
