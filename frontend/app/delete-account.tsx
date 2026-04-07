import { useState } from "react";
import { View, Text, TextInput, TouchableOpacity, ScrollView, Platform, Alert } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { router } from "expo-router";
import { useUser, useTheme, useTranslation } from "@/context/UserContext";

export default function DeleteAccount() {
  const { clearSession } = useUser();
  const theme = useTheme();
  const { t, isRTL } = useTranslation();

  const [step, setStep] = useState<1 | 2 | 3>(1);
  const [reason, setReason] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [confirmText, setConfirmText] = useState("");
  const [loading, setLoading] = useState(false);

  const REASONS = [
    "Je n'utilise plus l'application",
    "L'application ne répond pas à mes besoins",
    "J'ai des préoccupations concernant la confidentialité",
    "Je crée un nouveau compte",
    "Autre raison",
  ];

  const handleDelete = async () => {
    if (confirmText !== "SUPPRIMER") {
      Alert.alert("Confirmation requise", "Tapez SUPPRIMER pour confirmer.");
      return;
    }
    setLoading(true);
    // In production: call your backend API to delete the account here
    await new Promise(r => setTimeout(r, 2000));
    // Clear all local session data
    await clearSession();
    setLoading(false);
    // Navigate to login — replace() removes the entire back-stack
    router.replace("/(auth)/log-in");
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
          <Text style={{ fontSize: 20, fontFamily: "inter-semibold", color: theme.text }}>
            {t("deleteAccount")}
          </Text>
          <Text style={{ fontSize: 12, color: theme.danger, marginTop: 2 }}>Action irréversible</Text>
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
              <View style={{ flexDirection: "row", gap: 12, alignItems: "flex-start" }}>
                <Ionicons name="warning-outline" size={24} color={theme.danger} style={{ marginTop: 2 }} />
                <View style={{ flex: 1 }}>
                  <Text style={{ fontSize: 16, fontFamily: "inter-semibold", color: theme.danger, marginBottom: 8 }}>
                    Attention — Action définitive
                  </Text>
                  <Text style={{ fontSize: 13, color: theme.text, lineHeight: 20 }}>
                    La suppression de votre compte est permanente et irréversible. Toutes vos données, analyses, et historique seront définitivement effacés.
                  </Text>
                </View>
              </View>
            </View>

            {/* What will be deleted */}
            <View style={{ backgroundColor: theme.card, borderRadius: 18, padding: 16, gap: 10, shadowColor: theme.shadow, shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.05, elevation: 2 }}>
              <Text style={{ fontSize: 14, fontFamily: "inter-semibold", color: theme.text, marginBottom: 4 }}>Ce qui sera supprimé :</Text>
              {["Votre profil et informations personnelles", "Toutes vos analyses de contrats", "Votre historique de conversations", "Vos préférences et paramètres"].map((item, i) => (
                <View key={i} style={{ flexDirection: "row", alignItems: "center", gap: 10 }}>
                  <Ionicons name="close-circle-outline" size={16} color={theme.danger} />
                  <Text style={{ fontSize: 13, color: theme.textSecondary }}>{item}</Text>
                </View>
              ))}
            </View>

            {/* Reason */}
            <View style={{ backgroundColor: theme.card, borderRadius: 18, padding: 16, gap: 2, shadowColor: theme.shadow, shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.05, elevation: 2 }}>
              <Text style={{ fontSize: 14, fontFamily: "inter-semibold", color: theme.text, marginBottom: 12 }}>
                Pourquoi souhaitez-vous supprimer votre compte ?
              </Text>
              {REASONS.map((r) => (
                <TouchableOpacity
                  key={r}
                  onPress={() => setReason(r)}
                  style={{ flexDirection: "row", alignItems: "center", paddingVertical: 12, borderBottomWidth: 0.5, borderBottomColor: theme.divider, gap: 12 }}
                >
                  <View style={{
                    width: 20, height: 20, borderRadius: 10,
                    borderWidth: 2, borderColor: reason === r ? theme.danger : theme.border,
                    alignItems: "center", justifyContent: "center",
                  }}>
                    {reason === r && <View style={{ width: 10, height: 10, borderRadius: 5, backgroundColor: theme.danger }} />}
                  </View>
                  <Text style={{ flex: 1, fontSize: 14, color: theme.text }}>{r}</Text>
                </TouchableOpacity>
              ))}
            </View>

            <TouchableOpacity
              onPress={() => { if (!reason) { Alert.alert("Raison requise", "Veuillez sélectionner une raison."); return; } setStep(2); }}
              style={{ backgroundColor: theme.danger, borderRadius: 16, paddingVertical: 18, alignItems: "center" }}
            >
              <Text style={{ fontSize: 15, fontFamily: "inter-semibold", color: "#fff" }}>Continuer</Text>
            </TouchableOpacity>
          </>
        )}

        {/* Step 2: Password Confirmation */}
        {step === 2 && (
          <>
            <View style={{ backgroundColor: theme.card, borderRadius: 20, padding: 20, gap: 16, shadowColor: theme.shadow, shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.06, elevation: 2 }}>
              <Text style={{ fontSize: 16, fontFamily: "inter-semibold", color: theme.text }}>
                Confirmez votre identité
              </Text>
              <Text style={{ fontSize: 13, color: theme.textSecondary, lineHeight: 20 }}>
                Entrez votre mot de passe pour confirmer que vous êtes bien le propriétaire de ce compte.
              </Text>
              <View>
                <Text style={{ fontSize: 12, fontFamily: "inter-semibold", color: theme.textSecondary, marginBottom: 8, textTransform: "uppercase", letterSpacing: 0.5 }}>
                  Mot de passe
                </Text>
                <View style={{ flexDirection: "row", alignItems: "center", backgroundColor: theme.inputBg, borderRadius: 14, paddingHorizontal: 14, height: 52, borderWidth: 1, borderColor: theme.border }}>
                  <Ionicons name="lock-closed-outline" size={18} color={theme.textMuted} style={{ marginRight: 10 }} />
                  <TextInput
                    value={password}
                    onChangeText={setPassword}
                    placeholder="Votre mot de passe"
                    placeholderTextColor={theme.textMuted}
                    secureTextEntry={!showPassword}
                    style={{ flex: 1, fontSize: 15, color: theme.text }}
                  />
                  <TouchableOpacity onPress={() => setShowPassword(v => !v)}>
                    <Ionicons name={showPassword ? "eye-off-outline" : "eye-outline"} size={20} color={theme.textMuted} />
                  </TouchableOpacity>
                </View>
              </View>
            </View>
            <View style={{ flexDirection: "row", gap: 12 }}>
              <TouchableOpacity onPress={() => setStep(1)} style={{ flex: 1, backgroundColor: theme.pillBg, borderRadius: 16, paddingVertical: 16, alignItems: "center" }}>
                <Text style={{ fontSize: 15, fontFamily: "inter-semibold", color: theme.textSecondary }}>Retour</Text>
              </TouchableOpacity>
              <TouchableOpacity onPress={() => { if (!password) { Alert.alert("Mot de passe requis"); return; } setStep(3); }} style={{ flex: 2, backgroundColor: theme.danger, borderRadius: 16, paddingVertical: 16, alignItems: "center" }}>
                <Text style={{ fontSize: 15, fontFamily: "inter-semibold", color: "#fff" }}>Continuer</Text>
              </TouchableOpacity>
            </View>
          </>
        )}

        {/* Step 3: Final Confirmation */}
        {step === 3 && (
          <>
            <View style={{ backgroundColor: theme.card, borderRadius: 20, padding: 20, gap: 16, shadowColor: theme.shadow, shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.06, elevation: 2 }}>
              <Text style={{ fontSize: 16, fontFamily: "inter-semibold", color: theme.danger, textAlign: "center" }}>
                Dernière confirmation
              </Text>
              <Text style={{ fontSize: 13, color: theme.textSecondary, textAlign: "center", lineHeight: 20 }}>
                Tapez <Text style={{ fontFamily: "inter-semibold", color: theme.danger }}>SUPPRIMER</Text> pour confirmer définitivement la suppression de votre compte.
              </Text>
              <View style={{ backgroundColor: theme.inputBg, borderRadius: 14, paddingHorizontal: 14, height: 52, borderWidth: 1, borderColor: confirmText === "SUPPRIMER" ? theme.danger : theme.border, justifyContent: "center" }}>
                <TextInput
                  value={confirmText}
                  onChangeText={setConfirmText}
                  placeholder="SUPPRIMER"
                  placeholderTextColor={theme.textMuted}
                  autoCapitalize="characters"
                  style={{ fontSize: 16, color: theme.danger, fontFamily: "inter-semibold", textAlign: "center", letterSpacing: 2 }}
                />
              </View>
            </View>
            <View style={{ flexDirection: "row", gap: 12 }}>
              <TouchableOpacity onPress={() => setStep(2)} style={{ flex: 1, backgroundColor: theme.pillBg, borderRadius: 16, paddingVertical: 16, alignItems: "center" }}>
                <Text style={{ fontSize: 15, fontFamily: "inter-semibold", color: theme.textSecondary }}>Annuler</Text>
              </TouchableOpacity>
              <TouchableOpacity onPress={handleDelete} disabled={loading} style={{ flex: 2, backgroundColor: confirmText === "SUPPRIMER" ? theme.danger : theme.border, borderRadius: 16, paddingVertical: 16, alignItems: "center" }}>
                <Text style={{ fontSize: 15, fontFamily: "inter-semibold", color: "#fff" }}>
                  {loading ? "Suppression…" : "Supprimer définitivement"}
                </Text>
              </TouchableOpacity>
            </View>
          </>
        )}
      </ScrollView>
    </View>
  );
}
