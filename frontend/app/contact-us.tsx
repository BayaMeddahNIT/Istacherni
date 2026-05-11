import { useState, useEffect } from "react";
import { View, Text, ScrollView, TouchableOpacity, TextInput, Platform, Alert, KeyboardAvoidingView, TouchableWithoutFeedback, Keyboard } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { router } from "expo-router";
import { useUser, useTheme, useTranslation } from "@/context/UserContext";
import { apiContact, apiFetch } from "@/services/api";

export default function ContactUs() {
  const { user } = useUser();
  const theme = useTheme();
  const { t, isRTL } = useTranslation();

  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [subject, setSubject] = useState("");
  const [message, setMessage] = useState("");
  const [sending, setSending] = useState(false);
  const [sent, setSent] = useState(false);

  useEffect(() => {
    let active = true;
    const prefillUserData = async () => {
      // 1. Try from UserContext
      if (user && (user.name || user.email)) {
        if (user.name && !name) setName(user.name);
        if (user.email && !email) setEmail(user.email);
        return;
      }

      // 2. Fallback to API /profile
      try {
        const res = await apiFetch("/profile");
        if (res.ok) {
          const data = await res.json();
          if (active) {
            if (data.name && !name) setName(data.name);
            if (data.email && !email) setEmail(data.email);
          }
        }
      } catch (err) {
        console.log("Failed to fetch profile safely:", err);
      }
    };

    prefillUserData();
    return () => { active = false; };
  }, [user]);

  const handleSend = async () => {
    if (!name.trim() || !email.trim() || !message.trim()) {
      Alert.alert(t("allFieldsRequired"), t("fillAllFields"));
      return;
    }
    setSending(true);
    try {
      await apiContact(name, email, subject, message);
      setSent(true);
      setSubject("");
      setMessage("");
      setTimeout(() => {
        setSent(false);
      }, 4000);
    } catch (error: any) {
      Alert.alert(t("errorTitle"), error.message || t("failedToSendMessage"));
    } finally {
      setSending(false);
    }
  };

  return (
    <View style={{ flex: 1, backgroundColor: theme.background }}>
      {sent && (
        <View style={{
          position: "absolute",
          top: Platform.OS === "ios" ? 110 : 90,
          left: 20,
          right: 20,
          backgroundColor: theme.success,
          borderRadius: 12,
          padding: 16,
          flexDirection: isRTL ? "row-reverse" : "row",
          alignItems: "center",
          gap: 12,
          zIndex: 999,
          shadowColor: theme.shadow,
          shadowOffset: { width: 0, height: 4 },
          shadowOpacity: 0.15,
          shadowRadius: 8,
          elevation: 5,
        }}>
          <Ionicons name="checkmark-circle" size={24} color="#fff" />
          <Text style={{ flex: 1, color: "#fff", fontSize: 14, fontFamily: "inter-semibold", textAlign: isRTL ? "right" : "left" }}>
            {t("messageSent")}
          </Text>
        </View>
      )}
      <View style={{
        paddingTop: Platform.OS === "ios" ? 56 : 44, paddingBottom: 20, paddingHorizontal: 20,
        backgroundColor: theme.headerBg, borderBottomLeftRadius: 28, borderBottomRightRadius: 28,
        flexDirection: isRTL ? "row-reverse" : "row", alignItems: "center", gap: 14,
      }}>
        <TouchableOpacity onPress={() => router.back()} style={{ width: 40, height: 40, borderRadius: 12, backgroundColor: "rgba(255,255,255,0.3)", alignItems: "center", justifyContent: "center" }}>
          <Ionicons name={isRTL ? "arrow-forward" : "arrow-back"} size={22} color={theme.text} />
        </TouchableOpacity>
        <View style={{ flex: 1 }}>
          <Text style={{ fontSize: 20, fontFamily: "inter-semibold", color: theme.text, textAlign: isRTL ? "right" : "left" }}>{t("contactTitle")}</Text>
          <Text style={{ fontSize: 12, color: theme.primary, marginTop: 2, textAlign: isRTL ? "right" : "left" }}>{t("respondIn24h")}</Text>
        </View>
      </View>

      <KeyboardAvoidingView
        behavior={Platform.OS === "ios" ? "padding" : "height"}
        style={{ flex: 1 }}
        keyboardVerticalOffset={Platform.OS === "ios" ? 0 : 20}
      >
        <ScrollView contentContainerStyle={{ flexGrow: 1, padding: 20, gap: 20 }} showsVerticalScrollIndicator={false} keyboardShouldPersistTaps="handled">
          <TouchableWithoutFeedback onPress={Keyboard.dismiss}>
            <View style={{ gap: 20, flex: 1 }}>
              {/* Info Cards */}
              <View style={{ flexDirection: "row", gap: 12 }}>
                {[
                  { icon: "mail-outline", label: t("email"), value: "Istacherni@gmail.com" },
                  { icon: "call-outline", label: t("phoneField"), value: "+213 54 11 31 33" },
                ].map((item) => (
                  <View key={item.label} style={{
                    flex: 1, backgroundColor: theme.card, borderRadius: 16, padding: 16, alignItems: "center", gap: 8,
                    shadowColor: theme.shadow, shadowOffset: { width: 0, height: 1 }, shadowOpacity: 0.05, elevation: 1,
                  }}>
                    <View style={{ width: 40, height: 40, borderRadius: 12, backgroundColor: theme.primaryLight, alignItems: "center", justifyContent: "center" }}>
                      <Ionicons name={item.icon as any} size={20} color={theme.primary} />
                    </View>
                    <Text style={{ fontSize: 11, fontFamily: "inter-medium", color: theme.textSecondary }}>{item.label}</Text>
                    <Text style={{ fontSize: 11, fontFamily: "inter-semibold", color: theme.text, textAlign: "center" }}>{item.value}</Text>
                  </View>
                ))}
              </View>

              {/* Form */}
              <View style={{ backgroundColor: theme.card, borderRadius: 20, padding: 20, gap: 18, shadowColor: theme.shadow, shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.06, elevation: 2 }}>
                <Text style={{ fontSize: 15, fontFamily: "inter-semibold", color: theme.text, textAlign: isRTL ? "right" : "left" }}>
                  {t("sendAMessage")}
                </Text>

                {[
                  { label: t("fullName") + " *", value: name, setter: setName, icon: "person-outline", placeholder: t("namePlaceholder"), keyboard: "default" },
                  { label: t("emailAddress") + " *", value: email, setter: setEmail, icon: "mail-outline", placeholder: t("emailPlaceholder"), keyboard: "email-address" },
                  { label: t("subject"), value: subject, setter: setSubject, icon: "chatbox-outline", placeholder: t("subjectPlaceholder"), keyboard: "default" },
                ].map((field) => (
                  <View key={field.label}>
                    <Text style={{ fontSize: 12, fontFamily: "inter-semibold", color: theme.textSecondary, marginBottom: 8, textTransform: "uppercase", letterSpacing: 0.5, textAlign: isRTL ? "right" : "left" }}>
                      {field.label}
                    </Text>
                    <View style={{ flexDirection: isRTL ? "row-reverse" : "row", alignItems: "center", backgroundColor: theme.inputBg, borderRadius: 14, paddingHorizontal: 14, height: 50, borderWidth: 1, borderColor: theme.border }}>
                      <Ionicons name={field.icon as any} size={18} color={theme.textMuted} style={{ marginRight: isRTL ? 0 : 10, marginLeft: isRTL ? 10 : 0 }} />
                      <TextInput value={field.value} onChangeText={field.setter} placeholder={field.placeholder} placeholderTextColor={theme.textMuted} keyboardType={field.keyboard as any} autoCapitalize="none" style={{ flex: 1, fontSize: 15, color: theme.text, textAlign: isRTL ? "right" : "left" }} />
                    </View>
                  </View>
                ))}

                <View>
                  <Text style={{ fontSize: 12, fontFamily: "inter-semibold", color: theme.textSecondary, marginBottom: 8, textTransform: "uppercase", letterSpacing: 0.5, textAlign: isRTL ? "right" : "left" }}>
                    {t("message")} *
                  </Text>
                  <View style={{ backgroundColor: theme.inputBg, borderRadius: 14, padding: 14, borderWidth: 1, borderColor: theme.border, minHeight: 120 }}>
                    <TextInput value={message} onChangeText={setMessage} placeholder={t("messagePlaceholder")} placeholderTextColor={theme.textMuted} multiline textAlignVertical="top" style={{ fontSize: 15, color: theme.text, minHeight: 100, textAlign: isRTL ? "right" : "left" }} />
                  </View>
                </View>

                <TouchableOpacity onPress={handleSend} disabled={sending} activeOpacity={0.88} style={{ backgroundColor: sending ? theme.textMuted : theme.primary, borderRadius: 16, paddingVertical: 16, flexDirection: "row", alignItems: "center", justifyContent: "center", gap: 10 }}>
                  <Ionicons name={sending ? "hourglass-outline" : "send"} size={18} color="#fff" />
                  <Text style={{ fontSize: 15, fontFamily: "inter-semibold", color: "#fff" }}>
                    {sending ? t("sending") : t("send")}
                  </Text>
                </TouchableOpacity>
              </View>
            </View>
          </TouchableWithoutFeedback>
        </ScrollView>
      </KeyboardAvoidingView>
    </View>
  );
}
