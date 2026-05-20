import { useState } from "react";
import { View, Text, TextInput, TouchableOpacity, ScrollView, Platform, Alert } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { router } from "expo-router";
import { useTheme, useTranslation } from "@/context/UserContext";

export default function ChangePassword() {
  const theme = useTheme();
  const { t, isRTL } = useTranslation();

  const [currentPwd, setCurrentPwd] = useState("");
  const [newPwd, setNewPwd] = useState("");
  const [confirmPwd, setConfirmPwd] = useState("");
  const [showCurrent, setShowCurrent] = useState(false);
  const [showNew, setShowNew] = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);
  const [loading, setLoading] = useState(false);
  const [done, setDone] = useState(false);

  const strength = newPwd.length === 0 ? 0 : newPwd.length < 6 ? 1 : newPwd.length < 10 ? 2 : 3;
  const strengthColors = ["transparent", theme.danger, "#F39C12", theme.success];
  const strengthLabels = ["", t("pwdWeak"), t("pwdMedium"), t("pwdStrong")];

  const handleSubmit = async () => {
    if (!currentPwd || !newPwd || !confirmPwd) {
      Alert.alert(t("errorReqFields"), t("errorReqFields"));
      return;
    }
    if (newPwd !== confirmPwd) {
      Alert.alert(t("errorTitle"), t("pwdNoMatch"));
      return;
    }
    if (newPwd.length < 8) {
      Alert.alert(t("errorTooShort"), t("errorShortPwd"));
      return;
    }
    setLoading(true);
    await new Promise(r => setTimeout(r, 1600));
    setLoading(false);
    setDone(true);
    setTimeout(() => router.back(), 1800);
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
            {t("changePassword")}
          </Text>
          <Text style={{ fontSize: 12, color: theme.primary, marginTop: 2, textAlign: isRTL ? "right" : "left" }}>
            {t("changePwdSubtitle")}
          </Text>
        </View>
      </View>

      <ScrollView contentContainerStyle={{ padding: 24, gap: 20 }} showsVerticalScrollIndicator={false}>
        {done ? (
          <View style={{ alignItems: "center", paddingVertical: 40 }}>
            <View style={{ width: 80, height: 80, borderRadius: 40, backgroundColor: theme.success + "20", alignItems: "center", justifyContent: "center", marginBottom: 20 }}>
              <Ionicons name="checkmark-circle" size={48} color={theme.success} />
            </View>
            <Text style={{ fontSize: 20, fontFamily: "inter-semibold", color: theme.text, textAlign: "center" }}>
              {t("pwdChanged")}
            </Text>
            <Text style={{ fontSize: 14, color: theme.textSecondary, textAlign: "center", marginTop: 8 }}>
              {t("pwdChangedDesc")}
            </Text>
          </View>
        ) : (
          <>
            {/* Info Banner */}
            <View style={{ backgroundColor: theme.primaryLight, borderRadius: 16, padding: 16, flexDirection: "row", gap: 12, alignItems: "flex-start", borderWidth: 1, borderColor: theme.primaryMedium }}>
              <Ionicons name="lock-closed-outline" size={20} color={theme.primary} style={{ marginTop: 1 }} />
              <Text style={{ flex: 1, fontSize: 13, color: theme.primary, lineHeight: 20, textAlign: isRTL ? "right" : "left" }}>
                {t("pwdStrongReq")}
              </Text>
            </View>

            {/* Fields */}
            <View style={{ backgroundColor: theme.card, borderRadius: 20, padding: 20, gap: 20, shadowColor: theme.shadow, shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.06, elevation: 2 }}>
              {[
                { label: t("currentPwd"), value: currentPwd, setter: setCurrentPwd, show: showCurrent, toggle: () => setShowCurrent(v => !v) },
                { label: t("newPwd"), value: newPwd, setter: setNewPwd, show: showNew, toggle: () => setShowNew(v => !v) },
                { label: t("confirmPwd"), value: confirmPwd, setter: setConfirmPwd, show: showConfirm, toggle: () => setShowConfirm(v => !v) },
              ].map((field, i) => (
                <View key={i}>
                  <Text style={{ fontSize: 12, fontFamily: "inter-semibold", color: theme.textSecondary, marginBottom: 8, textTransform: "uppercase", letterSpacing: 0.5, textAlign: isRTL ? "right" : "left" }}>
                    {field.label}
                  </Text>
                  <View style={{ flexDirection: isRTL ? "row-reverse" : "row", alignItems: "center", backgroundColor: theme.inputBg, borderRadius: 14, paddingHorizontal: 14, height: 52, borderWidth: 1, borderColor: theme.border }}>
                    <Ionicons name="lock-closed-outline" size={18} color={theme.textMuted} style={{ marginRight: isRTL ? 0 : 10, marginLeft: isRTL ? 10 : 0 }} />
                    <TextInput
                      value={field.value}
                      onChangeText={field.setter}
                      placeholder="••••••••"
                      placeholderTextColor={theme.textMuted}
                      secureTextEntry={!field.show}
                      style={{ flex: 1, fontSize: 16, color: theme.text, letterSpacing: field.show ? 0 : 2, textAlign: isRTL ? "right" : "left" }}
                    />
                    <TouchableOpacity onPress={field.toggle} hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}>
                      <Ionicons name={field.show ? "eye-off-outline" : "eye-outline"} size={20} color={theme.textMuted} />
                    </TouchableOpacity>
                  </View>
                  {/* Strength bar for new password */}
                  {i === 1 && newPwd.length > 0 && (
                    <View style={{ marginTop: 10 }}>
                      <View style={{ flexDirection: "row", gap: 4, marginBottom: 4 }}>
                        {[1, 2, 3].map((s) => (
                          <View key={s} style={{ flex: 1, height: 4, borderRadius: 2, backgroundColor: s <= strength ? strengthColors[strength] : theme.border }} />
                        ))}
                      </View>
                      <Text style={{ fontSize: 11, color: strengthColors[strength], fontFamily: "inter-medium", textAlign: isRTL ? "right" : "left" }}>
                        {t("pwdStrength")} {strengthLabels[strength]}
                      </Text>
                    </View>
                  )}
                  {/* Match indicator for confirm */}
                  {i === 2 && confirmPwd.length > 0 && (
                    <Text style={{ fontSize: 11, marginTop: 6, color: confirmPwd === newPwd ? theme.success : theme.danger, fontFamily: "inter-medium", textAlign: isRTL ? "right" : "left" }}>
                      {confirmPwd === newPwd ? t("pwdMatch") : t("pwdNoMatch")}
                    </Text>
                  )}
                </View>
              ))}
            </View>

            {/* Submit */}
            <TouchableOpacity
              onPress={handleSubmit}
              disabled={loading}
              activeOpacity={0.88}
              style={{ backgroundColor: loading ? theme.textMuted : theme.primary, borderRadius: 16, paddingVertical: 18, flexDirection: "row", alignItems: "center", justifyContent: "center", gap: 10 }}
            >
              <Ionicons name={loading ? "hourglass-outline" : "checkmark-circle-outline"} size={20} color="#fff" />
              <Text style={{ fontSize: 16, fontFamily: "inter-semibold", color: "#fff" }}>
                {loading ? t("changingPwd") : t("changePwdBtn")}
              </Text>
            </TouchableOpacity>
          </>
        )}
      </ScrollView>
    </View>
  );
}
