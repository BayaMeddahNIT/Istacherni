import { useState } from "react";
import { View, Text, TouchableOpacity, ScrollView, Switch, Platform, Alert, Linking } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { router } from "expo-router";
import { useTheme, useTranslation } from "@/context/UserContext";

export default function PrivacySecurity() {
  const theme = useTheme();
  const { t, isRTL } = useTranslation();

  const [biometrics, setBiometrics] = useState(false);
  const [twoFactor, setTwoFactor] = useState(false);
  const [activityLog, setActivityLog] = useState(true);
  const [dataSharing, setDataSharing] = useState(false);

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
            {t("privacySecurity")}
          </Text>
          <Text style={{ fontSize: 12, color: theme.primary, marginTop: 2, textAlign: isRTL ? "right" : "left" }}>
            Gérez vos préférences de sécurité
          </Text>
        </View>
      </View>

      <ScrollView contentContainerStyle={{ padding: 20, gap: 24 }} showsVerticalScrollIndicator={false}>

        {/* Security Section */}
        <Section title="Sécurité du compte" theme={theme}>
          <ToggleRow icon="finger-print-outline" label="Authentification biométrique" subtitle="Utilisez votre empreinte ou Face ID" value={biometrics} onToggle={setBiometrics} theme={theme} isRTL={isRTL} />
          <View style={{ height: 0.5, backgroundColor: theme.divider }} />
          <ToggleRow icon="shield-checkmark-outline" label="Vérification en deux étapes" subtitle="Recevez un code par SMS lors de la connexion" value={twoFactor} onToggle={setTwoFactor} theme={theme} isRTL={isRTL} last />
        </Section>

        {/* Privacy Section */}
        <Section title="Confidentialité" theme={theme}>
          <ToggleRow icon="eye-outline" label="Journal d'activité" subtitle="Enregistrer les connexions et les actions" value={activityLog} onToggle={setActivityLog} theme={theme} isRTL={isRTL} />
          <View style={{ height: 0.5, backgroundColor: theme.divider }} />
          <ToggleRow icon="analytics-outline" label="Partage des données d'utilisation" subtitle="Aidez-nous à améliorer l'application (anonyme)" value={dataSharing} onToggle={setDataSharing} theme={theme} isRTL={isRTL} last />
        </Section>

        {/* Legal Links */}
        <Section title="Documents légaux" theme={theme}>
          {[
            { icon: "document-text-outline", label: "Politique de confidentialité", url: "https://istacherni.dz/privacy" },
            { icon: "shield-outline", label: "Conditions d'utilisation", url: "https://istacherni.dz/terms" },
            { icon: "information-circle-outline", label: "Loi n° 18-07 sur les données personnelles", url: "https://www.joradp.dz" },
          ].map((item, i, arr) => (
            <TouchableOpacity
              key={item.label}
              onPress={() => Linking.openURL(item.url)}
              style={{
                flexDirection: isRTL ? "row-reverse" : "row", alignItems: "center",
                paddingVertical: 14, paddingHorizontal: 4,
                borderBottomWidth: i < arr.length - 1 ? 0.5 : 0,
                borderBottomColor: theme.divider,
              }}
            >
              <View style={{ width: 36, height: 36, borderRadius: 10, backgroundColor: theme.primaryLight, alignItems: "center", justifyContent: "center", marginRight: isRTL ? 0 : 12, marginLeft: isRTL ? 12 : 0 }}>
                <Ionicons name={item.icon as any} size={18} color={theme.primary} />
              </View>
              <Text style={{ flex: 1, fontSize: 14, fontFamily: "inter-medium", color: theme.text, textAlign: isRTL ? "right" : "left" }}>
                {item.label}
              </Text>
              <Ionicons name="open-outline" size={16} color={theme.textMuted} />
            </TouchableOpacity>
          ))}
        </Section>

        {/* Active Sessions */}
        <Section title="Sessions actives" theme={theme}>
          {[
            { device: "iPhone 14 Pro", location: "Alger, DZ", time: "Maintenant", current: true },
            { device: "Chrome – Windows", location: "Alger, DZ", time: "Il y a 2 jours", current: false },
          ].map((session, i, arr) => (
            <View key={i} style={{
              flexDirection: isRTL ? "row-reverse" : "row", alignItems: "center",
              paddingVertical: 14, paddingHorizontal: 4,
              borderBottomWidth: i < arr.length - 1 ? 0.5 : 0,
              borderBottomColor: theme.divider,
            }}>
              <View style={{ width: 36, height: 36, borderRadius: 10, backgroundColor: theme.primaryLight, alignItems: "center", justifyContent: "center", marginRight: isRTL ? 0 : 12, marginLeft: isRTL ? 12 : 0 }}>
                <Ionicons name={session.current ? "phone-portrait-outline" : "desktop-outline"} size={18} color={theme.primary} />
              </View>
              <View style={{ flex: 1 }}>
                <View style={{ flexDirection: "row", alignItems: "center", gap: 8 }}>
                  <Text style={{ fontSize: 14, fontFamily: "inter-semibold", color: theme.text }}>{session.device}</Text>
                  {session.current && (
                    <View style={{ backgroundColor: theme.success + "20", borderRadius: 6, paddingHorizontal: 6, paddingVertical: 2 }}>
                      <Text style={{ fontSize: 10, color: theme.success, fontFamily: "inter-semibold" }}>Actuel</Text>
                    </View>
                  )}
                </View>
                <Text style={{ fontSize: 12, color: theme.textSecondary, marginTop: 2 }}>
                  {session.location} · {session.time}
                </Text>
              </View>
              {!session.current && (
                <TouchableOpacity onPress={() => Alert.alert("Session terminée", "Cette session a été déconnectée.")}>
                  <Ionicons name="close-circle-outline" size={22} color={theme.danger} />
                </TouchableOpacity>
              )}
            </View>
          ))}
        </Section>

      </ScrollView>
    </View>
  );
}

function Section({ title, children, theme }: { title: string; children: React.ReactNode; theme: any }) {
  return (
    <View>
      <Text style={{ fontSize: 12, fontFamily: "inter-semibold", color: theme.textSecondary, textTransform: "uppercase", letterSpacing: 0.8, marginBottom: 10, marginLeft: 4 }}>
        {title}
      </Text>
      <View style={{ backgroundColor: theme.card, borderRadius: 18, paddingHorizontal: 16, paddingVertical: 4, shadowColor: theme.shadow, shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.05, elevation: 2 }}>
        {children}
      </View>
    </View>
  );
}

function ToggleRow({ icon, label, subtitle, value, onToggle, theme, isRTL, last = false }: any) {
  return (
    <View style={{ flexDirection: isRTL ? "row-reverse" : "row", alignItems: "center", paddingVertical: 12 }}>
      <View style={{ width: 36, height: 36, borderRadius: 10, backgroundColor: theme.primaryLight, alignItems: "center", justifyContent: "center", marginRight: isRTL ? 0 : 12, marginLeft: isRTL ? 12 : 0 }}>
        <Ionicons name={icon} size={18} color={theme.primary} />
      </View>
      <View style={{ flex: 1 }}>
        <Text style={{ fontSize: 14, fontFamily: "inter-semibold", color: theme.text, textAlign: isRTL ? "right" : "left" }}>{label}</Text>
        <Text style={{ fontSize: 12, color: theme.textSecondary, marginTop: 2, textAlign: isRTL ? "right" : "left" }}>{subtitle}</Text>
      </View>
      <Switch value={value} onValueChange={onToggle} trackColor={{ false: theme.divider, true: theme.primary }} thumbColor={theme.card} ios_backgroundColor={theme.divider} />
    </View>
  );
}
