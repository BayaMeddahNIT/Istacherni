import { View, Text, ScrollView, TouchableOpacity, Switch, Platform, Alert } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { router } from "expo-router";
import { useUser, useTheme, useTranslation } from "@/context/UserContext";

const LANGUAGES = [
  { key: "fr" as const, label: "Français", flag: "🇫🇷", native: "Français" },
  { key: "ar" as const, label: "العربية", flag: "🇩🇿", native: "العربية" },
  { key: "en" as const, label: "English", flag: "🇬🇧", native: "English" },
];

export default function Settings() {
  const { language, setLanguage, darkMode, setDarkMode, notifications, setNotifications } = useUser();
  const theme = useTheme();
  const { t, isRTL } = useTranslation();

  const textAlign = isRTL ? "right" : "left";

  return (
    <View style={{ flex: 1, backgroundColor: theme.background }}>
      {/* Header */}
      <View style={{
        paddingTop: Platform.OS === "ios" ? 56 : 44,
        paddingBottom: 20, paddingHorizontal: 20,
        backgroundColor: theme.headerBg,
        borderBottomLeftRadius: 28, borderBottomRightRadius: 28,
        flexDirection: isRTL ? "row-reverse" : "row",
        alignItems: "center", gap: 14,
      }}>
        <TouchableOpacity onPress={() => router.back()} style={{
          width: 40, height: 40, borderRadius: 12,
          backgroundColor: "rgba(255,255,255,0.3)",
          alignItems: "center", justifyContent: "center",
        }}>
          <Ionicons name={isRTL ? "arrow-forward" : "arrow-back"} size={22} color={theme.text} />
        </TouchableOpacity>
        <Text style={{ fontSize: 20, fontFamily: "inter-semibold", color: theme.text }}>
          {t("settingsTitle")}
        </Text>
      </View>

      <ScrollView contentContainerStyle={{ padding: 20, gap: 24 }} showsVerticalScrollIndicator={false}>

        {/* ── Language ── */}
        <View>
          <SectionTitle label={t("appLanguage")} theme={theme} />
          <View style={cardStyle(theme)}>
            {LANGUAGES.map((lang, i) => {
              const active = language === lang.key;
              return (
                <TouchableOpacity
                  key={lang.key}
                  onPress={() => setLanguage(lang.key)}
                  activeOpacity={0.8}
                  style={{
                    flexDirection: isRTL ? "row-reverse" : "row",
                    alignItems: "center",
                    paddingVertical: 14, paddingHorizontal: 4,
                    borderBottomWidth: i < LANGUAGES.length - 1 ? 0.5 : 0,
                    borderBottomColor: theme.divider,
                  }}
                >
                  <Text style={{ fontSize: 24, marginRight: isRTL ? 0 : 14, marginLeft: isRTL ? 14 : 0 }}>
                    {lang.flag}
                  </Text>
                  <View style={{ flex: 1 }}>
                    <Text style={{
                      fontSize: 15,
                      fontFamily: active ? "inter-semibold" : "inter-regular",
                      color: active ? theme.primary : theme.text,
                      textAlign,
                    }}>
                      {lang.native}
                    </Text>
                  </View>
                  {active && <Ionicons name="checkmark-circle" size={22} color={theme.primary} />}
                </TouchableOpacity>
              );
            })}
          </View>
        </View>

        {/* ── Appearance ── */}
        <View>
          <SectionTitle label={t("appearance")} theme={theme} />
          <View style={cardStyle(theme)}>
            <View style={{ flexDirection: isRTL ? "row-reverse" : "row", alignItems: "center", paddingVertical: 8 }}>
              <View style={iconBox(theme)}>
                <Ionicons name={darkMode ? "moon" : "sunny-outline"} size={18} color={theme.primary} />
              </View>
              <View style={{ flex: 1, marginLeft: isRTL ? 0 : 12, marginRight: isRTL ? 12 : 0 }}>
                <Text style={{ fontSize: 15, fontFamily: "inter-semibold", color: theme.text, textAlign }}>
                  {t("darkMode")}
                </Text>
                <Text style={{ fontSize: 12, fontFamily: "inter-regular", color: theme.textSecondary, marginTop: 2, textAlign }}>
                  {darkMode ? t("darkModeOn") : t("darkModeOff")}
                </Text>
              </View>
              <Switch
                value={darkMode}
                onValueChange={setDarkMode}
                trackColor={{ false: theme.divider, true: theme.primary }}
                thumbColor={theme.card}
                ios_backgroundColor={theme.divider}
              />
            </View>
          </View>
        </View>

        {/* ── Notifications ── */}
        <View>
          <SectionTitle label={t("notifications")} theme={theme} />
          <View style={cardStyle(theme)}>
            <View style={{ flexDirection: isRTL ? "row-reverse" : "row", alignItems: "center", paddingVertical: 8 }}>
              <View style={iconBox(theme)}>
                <Ionicons name="notifications-outline" size={18} color={theme.primary} />
              </View>
              <View style={{ flex: 1, marginLeft: isRTL ? 0 : 12, marginRight: isRTL ? 12 : 0 }}>
                <Text style={{ fontSize: 15, fontFamily: "inter-semibold", color: theme.text, textAlign }}>
                  {t("pushNotifications")}
                </Text>
                <Text style={{ fontSize: 12, fontFamily: "inter-regular", color: theme.textSecondary, marginTop: 2, textAlign }}>
                  {t("alertsUpdates")}
                </Text>
              </View>
              <Switch
                value={notifications}
                onValueChange={setNotifications}
                trackColor={{ false: theme.divider, true: theme.primary }}
                thumbColor={theme.card}
                ios_backgroundColor={theme.divider}
              />
            </View>
          </View>
        </View>

        {/* ── Account ── */}
        <View>
          <SectionTitle label={t("account")} theme={theme} />
          <View style={cardStyle(theme)}>
            {[
              { icon: "lock-closed-outline", key: "changePassword", route: "/change-password", danger: false },
              { icon: "shield-checkmark-outline", key: "privacySecurity", route: "/privacy-security", danger: false },
              { icon: "trash-outline", key: "deleteAccount", route: "/delete-account", danger: true },
            ].map((item: any, i, arr) => (
              <TouchableOpacity
                key={item.key}
                onPress={() => router.push(item.route)}
                style={{
                  flexDirection: isRTL ? "row-reverse" : "row",
                  alignItems: "center",
                  paddingVertical: 14, paddingHorizontal: 4,
                  borderBottomWidth: i < arr.length - 1 ? 0.5 : 0,
                  borderBottomColor: theme.divider,
                }}
              >
                <View style={{
                  ...iconBox(theme),
                  backgroundColor: item.danger ? theme.danger + "14" : theme.primaryLight,
                }}>
                  <Ionicons name={item.icon} size={18} color={item.danger ? theme.danger : theme.primary} />
                </View>
                <Text style={{
                  flex: 1,
                  marginLeft: isRTL ? 0 : 12, marginRight: isRTL ? 12 : 0,
                  fontSize: 15, fontFamily: "inter-medium",
                  color: item.danger ? theme.danger : theme.text,
                  textAlign,
                }}>
                  {t(item.key as TranslationKeys)}
                </Text>
                <Ionicons name={isRTL ? "chevron-back" : "chevron-forward"} size={16} color={theme.textMuted} />
              </TouchableOpacity>
            ))}
          </View>
        </View>

        <Text style={{ textAlign: "center", fontSize: 12, color: theme.textMuted, fontFamily: "inter-regular" }}>
          {t("appVersion")}
        </Text>
      </ScrollView>
    </View>
  );
}

import { TranslationKeys } from "@/constants/i18n";
import type { Theme } from "@/constants/theme";

function SectionTitle({ label, theme }: { label: string; theme: Theme }) {
  return (
    <Text style={{
      fontSize: 12, fontFamily: "inter-semibold", color: theme.textSecondary,
      textTransform: "uppercase", letterSpacing: 0.8, marginBottom: 10, marginLeft: 4,
    }}>
      {label}
    </Text>
  );
}

function cardStyle(theme: Theme) {
  return {
    backgroundColor: theme.card,
    borderRadius: 18,
    paddingHorizontal: 16, paddingVertical: 4,
    shadowColor: theme.shadow,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.07, shadowRadius: 6, elevation: 2,
  };
}

function iconBox(theme: Theme) {
  return {
    width: 36, height: 36, borderRadius: 10,
    backgroundColor: theme.primaryLight,
    alignItems: "center" as const, justifyContent: "center" as const,
  };
}
