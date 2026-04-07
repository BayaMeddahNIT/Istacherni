import React, { createContext, useContext, useState, useEffect, ReactNode } from "react";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { lightTheme, darkTheme, Theme } from "@/constants/theme";
import translations, { LangKey, TranslationKeys } from "@/constants/i18n";

// ─── User Profile ─────────────────────────────────────────────────────────────

export interface UserProfile {
  name: string;
  email: string;
  phone: string;
  city: string;
  avatarUri: string | null;
}

interface UserContextType {
  // User data
  user: UserProfile;
  updateUser: (data: Partial<UserProfile>) => Promise<void>;
  clearSession: () => Promise<void>;

  // Language
  language: LangKey;
  setLanguage: (lang: LangKey) => Promise<void>;

  // Dark mode
  darkMode: boolean;
  setDarkMode: (val: boolean) => Promise<void>;

  // Notifications
  notifications: boolean;
  setNotifications: (val: boolean) => Promise<void>;

  // Derived: theme + translations
  theme: Theme;
  t: (key: TranslationKeys) => string;
  isRTL: boolean;
}

const defaultUser: UserProfile = {
  name: "",
  email: "",
  phone: "",
  city: "",
  avatarUri: null,
};

const UserContext = createContext<UserContextType | undefined>(undefined);

// ─── Provider ─────────────────────────────────────────────────────────────────

export function UserProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<UserProfile>(defaultUser);
  const [language, setLanguageState] = useState<LangKey>("fr");
  const [darkMode, setDarkModeState] = useState(false);
  const [notifications, setNotificationsState] = useState(true);

  useEffect(() => {
    loadPersistedData();
  }, []);

  const loadPersistedData = async () => {
    try {
      const [storedUser, storedLang, storedDark, storedNotif] = await Promise.all([
        AsyncStorage.getItem("@user_profile"),
        AsyncStorage.getItem("@language"),
        AsyncStorage.getItem("@dark_mode"),
        AsyncStorage.getItem("@notifications"),
      ]);
      if (storedUser) setUser(JSON.parse(storedUser));
      if (storedLang) setLanguageState(storedLang as LangKey);
      if (storedDark) setDarkModeState(storedDark === "true");
      if (storedNotif) setNotificationsState(storedNotif !== "false");
    } catch (e) {
      console.log("Error loading persisted data", e);
    }
  };

  const updateUser = async (data: Partial<UserProfile>) => {
    const newUser = { ...user, ...data };
    setUser(newUser);
    await AsyncStorage.setItem("@user_profile", JSON.stringify(newUser));
  };

  /**
   * Clears the user session: wipes AsyncStorage and resets all state.
   * Call this for both Logout and Delete Account.
   */
  const clearSession = async () => {
    try {
      await AsyncStorage.multiRemove([
        "@user_profile",
        // Keep language/dark-mode preferences across sessions
      ]);
    } catch (e) {
      console.log("Error clearing session", e);
    }
    setUser(defaultUser);
  };

  const setLanguage = async (lang: LangKey) => {
    setLanguageState(lang);
    await AsyncStorage.setItem("@language", lang);
  };

  const setDarkMode = async (val: boolean) => {
    setDarkModeState(val);
    await AsyncStorage.setItem("@dark_mode", String(val));
  };

  const setNotifications = async (val: boolean) => {
    setNotificationsState(val);
    await AsyncStorage.setItem("@notifications", String(val));
  };

  // Derived values — recomputed whenever darkMode or language changes
  const theme: Theme = darkMode ? darkTheme : lightTheme;
  const t = (key: TranslationKeys): string => translations[language][key] ?? translations.fr[key] ?? key;
  const isRTL = language === "ar";

  return (
    <UserContext.Provider value={{
      user, updateUser, clearSession,
      language, setLanguage,
      darkMode, setDarkMode,
      notifications, setNotifications,
      theme, t, isRTL,
    }}>
      {children}
    </UserContext.Provider>
  );
}

// ─── Hooks ────────────────────────────────────────────────────────────────────

export function useUser() {
  const ctx = useContext(UserContext);
  if (!ctx) throw new Error("useUser must be used within UserProvider");
  return ctx;
}

/** Shorthand: just get theme colors */
export function useTheme() {
  return useUser().theme;
}

/** Shorthand: just get translation function */
export function useTranslation() {
  const { t, language, isRTL } = useUser();
  return { t, language, isRTL };
}
