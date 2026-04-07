import { useState } from "react";
import { View, Text, ScrollView, TouchableOpacity, TextInput, Platform, Alert } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { router } from "expo-router";
import { useTheme, useTranslation } from "@/context/UserContext";

export default function ContactUs() {
  const theme = useTheme();
  const { t, isRTL } = useTranslation();

  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [subject, setSubject] = useState("");
  const [message, setMessage] = useState("");
  const [sending, setSending] = useState(false);
  const [sent, setSent] = useState(false);

  const handleSend = async () => {
    if (!name.trim() || !email.trim() || !message.trim()) {
      Alert.alert(t("allFieldsRequired"), t("fillAllFields"));
      return;
    }
    setSending(true);
    await new Promise(r => setTimeout(r, 1800));
    setSending(false);
    setSent(true);
  };

  if (sent) {
    return (
      <View style={{ flex: 1, backgroundColor: theme.background, alignItems: "center", justifyContent: "center", padding: 32 }}>
        <View style={{ width: 90, height: 90, borderRadius: 45, backgroundColor: theme.success + "20", alignItems: "center", justifyContent: "center", marginBottom: 24 }}>
          <Ionicons name="checkmark-circle" size={52} color={theme.success} />
        </View>
        <Text style={{ fontSize: 22, fontFamily: "inter-semibold", color: theme.text, textAlign: "center", marginBottom: 10 }}>
          {t("messageSent")}
        </Text>
        <Text style={{ fontSize: 15, color: theme.textSecondary, textAlign: "center", lineHeight: 22, marginBottom: 32 }}>
          {t("messageSentDesc")}
        </Text>
        <TouchableOpacity onPress={() => router.back()} style={{ backgroundColor: theme.primary, borderRadius: 16, paddingVertical: 16, paddingHorizontal: 40 }}>
          <Text style={{ fontSize: 15, fontFamily: "inter-semibold", color: "#fff" }}>{t("backToProfile")}</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={{ flex: 1, backgroundColor: theme.background }}>
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

      <ScrollView contentContainerStyle={{ padding: 20, gap: 20 }} showsVerticalScrollIndicator={false}>
        {/* Info Cards */}
        <View style={{ flexDirection: "row", gap: 12 }}>
          {[
            { icon: "mail-outline", label: t("email"), value: "contact@istacherni.dz" },
            { icon: "call-outline", label: t("phone"), value: "+213 XX XX XX XX" },
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
      </ScrollView>
    </View>
  );
}
