import React, { useState } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  Modal,
  StyleSheet,
  Pressable,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { useUser } from "@/context/UserContext";
import { LangKey } from "@/constants/i18n";

const languages: { key: LangKey; label: string; flag: string }[] = [
  { key: "ar", label: "العربية", flag: "🇩🇿" },
  { key: "fr", label: "Français", flag: "🇫🇷" },
  { key: "en", label: "English", flag: "🇺🇸" },
];

export default function LanguageSelector() {
  const { language, setLanguage, theme, isRTL } = useUser();
  const [modalVisible, setModalVisible] = useState(false);

  const handleSelect = async (lang: LangKey) => {
    await setLanguage(lang);
    setModalVisible(false);
  };

  const currentLang = languages.find((l) => l.key === language) || languages[1];

  return (
    <View style={styles.container}>
      <TouchableOpacity
        onPress={() => setModalVisible(true)}
        style={[
          styles.trigger,
          {
            backgroundColor: theme.primaryLight,
            borderColor: theme.primaryMedium,
            borderWidth: 1
          }
        ]}
        activeOpacity={0.7}
      >
        <Ionicons name="language-outline" size={18} color={theme.primary} />
        <Text style={[styles.triggerText, { color: theme.primary }]}>
          {currentLang.key.toUpperCase()}
        </Text>
      </TouchableOpacity>

      <Modal
        animationType="fade"
        transparent={true}
        visible={modalVisible}
        onRequestClose={() => setModalVisible(false)}
      >
        <Pressable
          style={styles.modalOverlay}
          onPress={() => setModalVisible(false)}
        >
          <View style={[styles.modalContent, { backgroundColor: "#DAD6D1", borderColor: "rgba(0,0,0,0.1)", borderWidth: 1 }]}>
            <Text style={[styles.modalTitle, { color: "#1A1A1A" }]}>
              {isRTL ? "اختر اللغة" : "Select Language"}
            </Text>
            {languages.map((lang) => (
              <TouchableOpacity
                key={lang.key}
                onPress={() => handleSelect(lang.key)}
                style={[
                  styles.langItem,
                  { borderBottomColor: "rgba(0,0,0,0.05)" },
                  language === lang.key && { backgroundColor: "rgba(255,255,255,0.3)" },
                ]}
              >
                <Text style={styles.flag}>{lang.flag}</Text>
                <Text
                  style={[
                    styles.langLabel,
                    { color: "#1A1A1A" },
                    language === lang.key && { color: theme.primary, fontWeight: "bold" },
                  ]}
                >
                  {lang.label}
                </Text>
                {language === lang.key && (
                  <Ionicons name="checkmark-circle" size={22} color={theme.primary} />
                )}
              </TouchableOpacity>
            ))}
          </View>
        </Pressable>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    zIndex: 1000,
  },
  trigger: {
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 20,
    borderWidth: 1,
    gap: 6,
  },
  triggerText: {
    fontSize: 14,
    fontWeight: "600",
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: "rgba(0,0,0,0.1)",
    justifyContent: "center",
    alignItems: "center",
  },
  modalContent: {
    width: "80%",
    borderRadius: 24,
    padding: 20,
    shadowColor: "#e77777ff",
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.1,
    shadowRadius: 20,
    elevation: 5,
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: "bold",
    marginBottom: 20,
    textAlign: "center",
  },
  langItem: {
    flexDirection: "row",
    alignItems: "center",
    paddingVertical: 16,
    borderBottomWidth: 1,
    paddingHorizontal: 12,
    borderRadius: 12,
    gap: 12,
  },
  flag: {
    fontSize: 22,
  },
  langLabel: {
    flex: 1,
    fontSize: 16,
  },
});
