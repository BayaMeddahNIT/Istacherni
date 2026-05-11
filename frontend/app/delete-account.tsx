import { useState } from "react";
import { View, Text, TextInput, TouchableOpacity, ScrollView, Platform, Alert } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { router } from "expo-router";
import { useUser, useTheme, useTranslation } from "@/context/UserContext";
import { useAuth } from "@/context/AuthContext";
import { apiDeleteAccount } from "@/services/api";

export default function DeleteAccount() {
  const { clearSession } = useUser();
  const { logout } = useAuth();
  const theme = useTheme();
  const { t, isRTL } = useTranslation();

  const [step, setStep] = useState<1 | 2 | 3>(1);
  const [reason, setReason] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [confirmText, setConfirmText] = useState("");
  const [loading, setLoading] = useState(false);

  const REASONS = [
    t("reasonNoLongerUse"),
    t("reasonNotMeetNeeds"),
    t("reasonPrivacy"),
    t("reasonNewAccount"),
    t("reasonOther"),
  ];

  const handleDelete = async () => {
    if (confirmText !== t("deleteWord")) {
      Alert.alert(t("errorTitle"), t("typeDeleteToConfirm"));
      return;
    }
    setLoading(true);
    try {
      await apiDeleteAccount();   // calls DELETE /auth/account with JWT
    } catch (e: any) {
      Alert.alert(t("errorTitle"), e.message || t("deleteFailed"));
      setLoading(false);
      return;
    }
    await clearSession();         // wipe UserContext profile
    await logout();               // wipe tokens + navigate to login
    setLoading(false);
  };

  return (
    <View style={{ flex: 1, backgroundColor: theme.background }}>
      {/* Header */}
      <View style={{
        paddingTop: Platform.OS === "ios" ? 56 : 44,
        paddingBottom: 20, paddingHorizontal: 20,
        backgroundColor: theme.headerBg,
        borderBottomLeftRadius: 28, borderBottomRightRadius: 28,
        flexDirection: isRTL ? "row-reverse" : "row", alignItems: "center", gap: 14,
      }}>
        <TouchableOpacity onPress={() => router.back()} style={{
          width: 40, height: 40, borderRadius: 12,
          backgroundColor: "rgba(255,255,255,0.3)", alignItems: "center", justifyContent: "center",
        }}>
          <Ionicons name={isRTL ? "arrow-forward" : "arrow-back"} size={22} color={theme.text} />
        </TouchableOpacity>
        <View style={{ flex: 1 }}>
          <Text style={{ fontSize: 20, fontFamily: "inter-semibold", color: theme.text, textAlign: isRTL ? "right" : "left" }}>
            {t("deleteAccount")}
          </Text>
          <Text style={{ fontSize: 12, color: theme.danger, marginTop: 2, textAlign: isRTL ? "right" : "left" }}>
            {t("deleteActionIrreversible")}
          </Text>
        </View>
      </View>

      {/* Step Indicator */}
      <View style={{ flexDirection: "row", paddingHorizontal: 24, paddingVertical: 16, gap: 8 }}>
        {[1, 2, 3].map((s) => (
          <View key={s} style={{ flex: 1, height: 4, borderRadius: 2, backgroundColor: s <= step ? theme.danger : theme.border }} />
        ))}
      </View>

      <ScrollView contentContainerStyle={{ padding: 24, gap: 20 }} showsVerticalScrollIndicator={false}>

        {/* Step 1: Warning + Reason */}
        {step === 1 && (
          <>
            <View style={{ backgroundColor: theme.danger + "12", borderRadius: 18, padding: 20, gap: 12, borderWidth: 1, borderColor: theme.danger + "25" }}>
              <View style={{ flexDirection: isRTL ? "row-reverse" : "row", gap: 12, alignItems: "flex-start" }}>
                <Ionicons name="warning-outline" size={24} color={theme.danger} style={{ marginTop: 2 }} />
                <View style={{ flex: 1 }}>
                  <Text style={{ fontSize: 16, fontFamily: "inter-semibold", color: theme.danger, marginBottom: 8, textAlign: isRTL ? "right" : "left" }}>
                    {t("warningPermanent")}
                  </Text>
                  <Text style={{ fontSize: 13, color: theme.text, lineHeight: 20, textAlign: isRTL ? "right" : "left" }}>
                    {t("warningPermanentDesc")}
                  </Text>
                </View>
              </View>
            </View>

            {/* What will be deleted */}
            <View style={{ backgroundColor: theme.card, borderRadius: 18, padding: 16, gap: 10, shadowColor: theme.shadow, shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.05, elevation: 2 }}>
              <Text style={{ fontSize: 14, fontFamily: "inter-semibold", color: theme.text, marginBottom: 4, textAlign: isRTL ? "right" : "left" }}>
                {t("whatWillBeDeleted")}
              </Text>
              {[t("deletedProfile"), t("deletedAnalyses"), t("deletedHistory"), t("deletedSettings")].map((item, i) => (
                <View key={i} style={{ flexDirection: isRTL ? "row-reverse" : "row", alignItems: "center", gap: 10 }}>
                  <Ionicons name="close-circle-outline" size={16} color={theme.danger} />
                  <Text style={{ fontSize: 13, color: theme.textSecondary, textAlign: isRTL ? "right" : "left" }}>{item}</Text>
                </View>
              ))}
            </View>

            {/* Reason */}
            <View style={{ backgroundColor: theme.card, borderRadius: 18, padding: 16, gap: 2, shadowColor: theme.shadow, shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.05, elevation: 2 }}>
              <Text style={{ fontSize: 14, fontFamily: "inter-semibold", color: theme.text, marginBottom: 12, textAlign: isRTL ? "right" : "left" }}>
                {t("whyDelete")}
              </Text>
              {REASONS.map((r) => (
                <TouchableOpacity
                  key={r}
                  onPress={() => setReason(r)}
                  style={{ flexDirection: isRTL ? "row-reverse" : "row", alignItems: "center", paddingVertical: 12, borderBottomWidth: 0.5, borderBottomColor: theme.divider, gap: 12 }}
                >
                  <View style={{
                    width: 20, height: 20, borderRadius: 10,
                    borderWidth: 2, borderColor: reason === r ? theme.danger : theme.border,
                    alignItems: "center", justifyContent: "center",
                  }}>
                    {reason === r && <View style={{ width: 10, height: 10, borderRadius: 5, backgroundColor: theme.danger }} />}
                  </View>
                  <Text style={{ flex: 1, fontSize: 14, color: theme.text, textAlign: isRTL ? "right" : "left" }}>{r}</Text>
                </TouchableOpacity>
              ))}
            </View>

            <TouchableOpacity
              onPress={() => { if (!reason) { Alert.alert(t("reasonRequired"), t("selectReason")); return; } setStep(2); }}
              style={{ backgroundColor: theme.danger, borderRadius: 16, paddingVertical: 18, alignItems: "center" }}
            >
              <Text style={{ fontSize: 15, fontFamily: "inter-semibold", color: "#fff" }}>{t("continue")}</Text>
            </TouchableOpacity>
          </>
        )}

        {/* Step 2: Password Confirmation */}
        {step === 2 && (
          <>
            <View style={{ backgroundColor: theme.card, borderRadius: 20, padding: 20, gap: 16, shadowColor: theme.shadow, shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.06, elevation: 2 }}>
              <Text style={{ fontSize: 16, fontFamily: "inter-semibold", color: theme.text, textAlign: isRTL ? "right" : "left" }}>
                {t("confirmIdentity")}
              </Text>
              <Text style={{ fontSize: 13, color: theme.textSecondary, lineHeight: 20, textAlign: isRTL ? "right" : "left" }}>
                {t("enterPwdToConfirm")}
              </Text>
              <View>
                <Text style={{ fontSize: 12, fontFamily: "inter-semibold", color: theme.textSecondary, marginBottom: 8, textTransform: "uppercase", letterSpacing: 0.5, textAlign: isRTL ? "right" : "left" }}>
                  {t("passwordLabel")}
                </Text>
                <View style={{ flexDirection: isRTL ? "row-reverse" : "row", alignItems: "center", backgroundColor: theme.inputBg, borderRadius: 14, paddingHorizontal: 14, height: 52, borderWidth: 1, borderColor: theme.border }}>
                  <Ionicons name="lock-closed-outline" size={18} color={theme.textMuted} style={{ marginRight: isRTL ? 0 : 10, marginLeft: isRTL ? 10 : 0 }} />
                  <TextInput
                    value={password}
                    onChangeText={setPassword}
                    placeholder={t("pwdPlaceholder")}
                    placeholderTextColor={theme.textMuted}
                    secureTextEntry={!showPassword}
                    style={{ flex: 1, fontSize: 15, color: theme.text, textAlign: isRTL ? "right" : "left" }}
                  />
                  <TouchableOpacity onPress={() => setShowPassword(v => !v)}>
                    <Ionicons name={showPassword ? "eye-off-outline" : "eye-outline"} size={20} color={theme.textMuted} />
                  </TouchableOpacity>
                </View>
              </View>
            </View>
            <View style={{ flexDirection: isRTL ? "row-reverse" : "row", gap: 12 }}>
              <TouchableOpacity onPress={() => setStep(1)} style={{ flex: 1, backgroundColor: theme.pillBg, borderRadius: 16, paddingVertical: 16, alignItems: "center" }}>
                <Text style={{ fontSize: 15, fontFamily: "inter-semibold", color: theme.textSecondary }}>{t("back")}</Text>
              </TouchableOpacity>
              <TouchableOpacity onPress={() => { if (!password) { Alert.alert(t("pwdRequired")); return; } setStep(3); }} style={{ flex: 2, backgroundColor: theme.danger, borderRadius: 16, paddingVertical: 16, alignItems: "center" }}>
                <Text style={{ fontSize: 15, fontFamily: "inter-semibold", color: "#fff" }}>{t("continue")}</Text>
              </TouchableOpacity>
            </View>
          </>
        )}

        {/* Step 3: Final Confirmation */}
        {step === 3 && (
          <>
            <View style={{ backgroundColor: theme.card, borderRadius: 20, padding: 20, gap: 16, shadowColor: theme.shadow, shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.06, elevation: 2 }}>
              <Text style={{ fontSize: 16, fontFamily: "inter-semibold", color: theme.danger, textAlign: "center" }}>
                {t("finalConfirmation")}
              </Text>
              <Text style={{ fontSize: 13, color: theme.textSecondary, textAlign: "center", lineHeight: 20 }}>
                {t("typeDeleteToConfirm")}
              </Text>
              <View style={{ backgroundColor: theme.inputBg, borderRadius: 14, paddingHorizontal: 14, height: 52, borderWidth: 1, borderColor: confirmText === t("deleteWord") ? theme.danger : theme.border, justifyContent: "center" }}>
                <TextInput
                  value={confirmText}
                  onChangeText={setConfirmText}
                  placeholder={t("deleteWord")}
                  placeholderTextColor={theme.textMuted}
                  autoCapitalize="characters"
                  style={{ fontSize: 16, color: theme.danger, fontFamily: "inter-semibold", textAlign: "center", letterSpacing: 2 }}
                />
              </View>
            </View>
            <View style={{ flexDirection: isRTL ? "row-reverse" : "row", gap: 12 }}>
              <TouchableOpacity onPress={() => setStep(2)} style={{ flex: 1, backgroundColor: theme.pillBg, borderRadius: 16, paddingVertical: 16, alignItems: "center" }}>
                <Text style={{ fontSize: 15, fontFamily: "inter-semibold", color: theme.textSecondary }}>{t("cancel")}</Text>
              </TouchableOpacity>
              <TouchableOpacity onPress={handleDelete} disabled={loading} style={{ flex: 2, backgroundColor: confirmText === t("deleteWord") ? theme.danger : theme.border, borderRadius: 16, paddingVertical: 16, alignItems: "center" }}>
                <Text style={{ fontSize: 15, fontFamily: "inter-semibold", color: "#fff" }}>
                  {loading ? t("deleting") : t("deletePermanently")}
                </Text>
              </TouchableOpacity>
            </View>
          </>
        )}
      </ScrollView>
    </View>
  );
}
