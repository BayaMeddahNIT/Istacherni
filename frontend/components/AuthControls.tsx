import React from "react";
import { View, Text, TouchableOpacity, StyleSheet, Platform } from "react-native";
import { useUser } from "@/context/UserContext";
import { Languages, Moon, Sun } from "lucide-react-native";
import { LangKey } from "@/constants/i18n";

export default function AuthControls() {
  const { theme, language, setLanguage, darkMode, setDarkMode, isRTL } = useUser();

  const toggleLanguage = () => {
    const langs: LangKey[] = ["fr", "ar", "en"];
    const next = langs[(langs.indexOf(language) + 1) % langs.length];
    setLanguage(next);
  };

  return (
    <View style={[
      styles.header, 
      { 
        flexDirection: isRTL ? 'row-reverse' : 'row',
        top: Platform.OS === 'ios' ? 50 : 20,
      }
    ]}>
      <TouchableOpacity 
        onPress={toggleLanguage} 
        style={styles.iconButton}
        activeOpacity={0.7}
      >
        <Languages size={22} color={theme.text} />
        <Text style={[styles.iconText, { color: theme.text }]}>
          {language.toUpperCase()}
        </Text>
      </TouchableOpacity>

      <TouchableOpacity 
        onPress={() => setDarkMode(!darkMode)} 
        style={styles.iconButton}
        activeOpacity={0.7}
      >
        {darkMode ? (
          <Sun size={24} color={theme.text} />
        ) : (
          <Moon size={22} color={theme.text} />
        )}
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  header: {
    position: "absolute",
    left: 0,
    right: 0,
    justifyContent: "space-between",
    alignItems: "center",
    paddingHorizontal: 20,
    zIndex: 1000,
  },
  iconButton: {
    flexDirection: "row",
    alignItems: "center",
    padding: 8,
    borderRadius: 20,
    backgroundColor: "rgba(255,255,255,0.05)",
  },
  iconText: {
    marginLeft: 6,
    fontWeight: "bold",
    fontSize: 14,
  },
});
