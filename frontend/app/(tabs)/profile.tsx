import {
  Text, View, TouchableOpacity, ScrollView, Image,
  Alert, Modal, Platform, Share,
} from "react-native";
import { useState } from "react";
import { Ionicons } from "@expo/vector-icons";
import { router } from "expo-router";
import { useUser, useTheme, useTranslation } from "@/context/UserContext";

function RateModal({ visible, onClose }: { visible: boolean; onClose: () => void }) {
  const theme = useTheme();
  const { t } = useTranslation();
  const [stars, setStars] = useState(0);
  const [submitted, setSubmitted] = useState(false);

  const handleSubmit = () => {
    if (stars === 0) { Alert.alert(t("selectRating")); return; }
    setSubmitted(true);
    setTimeout(() => { setSubmitted(false); setStars(0); onClose(); }, 1800);
  };

  return (
    <Modal visible={visible} transparent animationType="slide">
      <View style={{ flex: 1, backgroundColor: theme.overlay, justifyContent: "flex-end" }}>
        <View style={{
          backgroundColor: theme.card,
          borderTopLeftRadius: 28, borderTopRightRadius: 28, padding: 28,
        }}>
          <View style={{ width: 40, height: 4, backgroundColor: theme.divider, borderRadius: 2, alignSelf: "center", marginBottom: 24 }} />
          {submitted ? (
            <View style={{ alignItems: "center", paddingVertical: 20 }}>
              <Ionicons name="star" size={52} color="#F2C94C" />
              <Text style={{ fontSize: 20, fontFamily: "inter-semibold", color: theme.text, marginTop: 16 }}>
                {t("thankYou")}
              </Text>
            </View>
          ) : (
            <>
              <Text style={{ fontSize: 20, fontFamily: "inter-semibold", color: theme.text, textAlign: "center", marginBottom: 8 }}>
                {t("rateTitle")}
              </Text>
              <Text style={{ fontSize: 14, color: theme.textSecondary, textAlign: "center", marginBottom: 28 }}>
                {t("rateSubtitle")}
              </Text>
              <View style={{ flexDirection: "row", justifyContent: "center", gap: 12, marginBottom: 28 }}>
                {[1, 2, 3, 4, 5].map((s) => (
                  <TouchableOpacity key={s} onPress={() => setStars(s)}>
                    <Ionicons name={s <= stars ? "star" : "star-outline"} size={40} color={s <= stars ? "#F2C94C" : theme.divider} />
                  </TouchableOpacity>
                ))}
              </View>
              <TouchableOpacity onPress={handleSubmit} style={{
                backgroundColor: theme.primary, borderRadius: 16,
                paddingVertical: 16, alignItems: "center", marginBottom: 12,
              }}>
                <Text style={{ fontSize: 15, fontFamily: "inter-semibold", color: "#fff" }}>{t("submit")}</Text>
              </TouchableOpacity>
              <TouchableOpacity onPress={onClose} style={{ alignItems: "center", paddingVertical: 10 }}>
                <Text style={{ fontSize: 14, fontFamily: "inter-medium", color: theme.textSecondary }}>{t("later")}</Text>
              </TouchableOpacity>
            </>
          )}
        </View>
      </View>
    </Modal>
  );
}

function MenuItem({
  icon, label, subtitle, onPress, showBorder = true, danger = false,
}: {
  icon: string; label: string; subtitle?: string; onPress: () => void;
  showBorder?: boolean; danger?: boolean;
}) {
  const theme = useTheme();
  const { isRTL } = useTranslation();

  return (
    <TouchableOpacity
      onPress={onPress}
      activeOpacity={0.7}
      style={{
        flexDirection: isRTL ? "row-reverse" : "row",
        alignItems: "center",
        paddingVertical: 14, paddingHorizontal: 4,
        borderBottomWidth: showBorder ? 0.5 : 0,
        borderBottomColor: theme.divider,
      }}
    >
      <View style={{
        width: 40, height: 40, borderRadius: 12,
        backgroundColor: danger ? theme.danger + "14" : theme.primaryLight,
        alignItems: "center", justifyContent: "center", marginRight: isRTL ? 0 : 14, marginLeft: isRTL ? 14 : 0,
      }}>
        <Ionicons name={icon as any} size={20} color={danger ? theme.danger : theme.primary} />
      </View>
      <View style={{ flex: 1 }}>
        <Text style={{ fontSize: 15, fontFamily: "inter-medium", color: danger ? theme.danger : theme.text, textAlign: isRTL ? "right" : "left" }}>
          {label}
        </Text>
        {subtitle && (
          <Text style={{ fontSize: 12, color: theme.textSecondary, marginTop: 1, textAlign: isRTL ? "right" : "left" }}>
            {subtitle}
          </Text>
        )}
      </View>
      <Ionicons name={isRTL ? "chevron-back" : "chevron-forward"} size={16} color={theme.textMuted} />
    </TouchableOpacity>
  );
}

export default function Profile() {
  const { user, clearSession } = useUser();
  const theme = useTheme();
  const { t, isRTL } = useTranslation();
  const [rateVisible, setRateVisible] = useState(false);
  const [loggingOut, setLoggingOut] = useState(false);

  const handleLogout = () => {
    Alert.alert(
      t("logoutConfirmTitle"),
      t("logoutConfirmMsg"),
      [
        { text: t("cancel"), style: "cancel" },
        {
          text: t("logout"),
          style: "destructive",
          onPress: async () => {
            setLoggingOut(true);
            await clearSession();
            setLoggingOut(false);
            // Replace entire navigation stack — user cannot go back to home
            router.replace("/(auth)/log-in");
          },
        },
      ]
    );
  };

  const handleShare = async () => {
    try {
      await Share.share(
        {
          title: "Istacherni – Assistant Juridique Algérien",
          message:
            "Découvrez Istacherni, l'assistant juridique IA pour la loi algérienne 🇩🇿\n" +
            "Analysez vos contrats, trouvez un avocat et consultez la bibliothèque du droit algérien.\n\n" +
            "https://istacherni.dz",
          url: "https://istacherni.dz",
        },
        {
          dialogTitle: "Partager Istacherni",
          subject: "Découvrez Istacherni – Assistant Juridique IA",
        }
      );
    } catch (e) {
      console.log("Share error", e);
    }
  };

  const displayName = user.name && user.name.trim().length > 0 ? user.name : null;

  return (
    <View style={{ flex: 1, backgroundColor: theme.background }}>
      <ScrollView contentContainerStyle={{ paddingBottom: 40 }} showsVerticalScrollIndicator={false}>

        {/* Header */}
        <View style={{
          paddingTop: Platform.OS === "ios" ? 60 : 48,
          paddingBottom: 32, paddingHorizontal: 24,
          backgroundColor: theme.headerBg,
          borderBottomLeftRadius: 30, borderBottomRightRadius: 30,
        }}>
          <View style={{ flexDirection: isRTL ? "row-reverse" : "row", alignItems: "center" }}>
            {/* Avatar */}
            <TouchableOpacity onPress={() => router.push("/profile-edit")} activeOpacity={0.85}>
              <View style={{
                width: 72, height: 72, borderRadius: 36,
                backgroundColor: theme.primary,
                alignItems: "center", justifyContent: "center",
                borderWidth: 3, borderColor: theme.card,
                shadowColor: theme.shadow, shadowOffset: { width: 0, height: 3 },
                shadowOpacity: 0.15, shadowRadius: 8, elevation: 6, overflow: "hidden",
              }}>
                {user.avatarUri
                  ? <Image source={{ uri: user.avatarUri }} style={{ width: 72, height: 72 }} />
                  : <Ionicons name="person" size={34} color="#FFFFFF" />
                }
              </View>
              <View style={{
                position: "absolute", bottom: 0, right: isRTL ? undefined : 0, left: isRTL ? 0 : undefined,
                backgroundColor: theme.primary, borderRadius: 10,
                width: 22, height: 22, alignItems: "center", justifyContent: "center",
                borderWidth: 2, borderColor: theme.headerBg,
              }}>
                <Ionicons name="pencil" size={10} color="#fff" />
              </View>
            </TouchableOpacity>

            {/* Name */}
            <View style={{ marginLeft: isRTL ? 0 : 16, marginRight: isRTL ? 16 : 0, flex: 1 }}>
              <Text style={{ fontSize: 13, color: theme.primary, fontFamily: "inter-regular", marginBottom: 2, textAlign: isRTL ? "right" : "left" }}>
                {t("welcomeBack")}
              </Text>
              <Text style={{ fontSize: 22, color: theme.text, fontFamily: "inter-semibold", textAlign: isRTL ? "right" : "left" }}>
                {displayName ? `${t("hello")}, ${displayName.split(" ")[0]}` : t("myAccount")}
              </Text>
              {user.email ? (
                <Text style={{ fontSize: 12, color: theme.textSecondary, marginTop: 2, textAlign: isRTL ? "right" : "left" }}>
                  {user.email}
                </Text>
              ) : null}
            </View>

            <TouchableOpacity style={{
              width: 42, height: 42, borderRadius: 13,
              backgroundColor: "rgba(255,255,255,0.25)",
              alignItems: "center", justifyContent: "center",
            }}>
              <Ionicons name="notifications-outline" size={22} color={theme.primary} />
            </TouchableOpacity>
          </View>

          {/* Stats */}
          {displayName && (
            <View style={{ flexDirection: "row", marginTop: 20, gap: 10 }}>
              {[
                { label: t("analyses"), value: "4" },
                { label: t("contracts"), value: "2" },
                { label: t("consultations"), value: "7" },
              ].map((stat) => (
                <View key={stat.label} style={{
                  flex: 1, backgroundColor: "rgba(255,255,255,0.2)",
                  borderRadius: 14, padding: 12, alignItems: "center",
                }}>
                  <Text style={{ fontSize: 18, fontFamily: "inter-semibold", color: theme.text }}>{stat.value}</Text>
                  <Text style={{ fontSize: 11, color: theme.primary, marginTop: 2, textAlign: "center" }}>{stat.label}</Text>
                </View>
              ))}
            </View>
          )}
        </View>

        {/* Profile Card */}
        <View style={{ paddingHorizontal: 20, marginTop: -16 }}>
          <TouchableOpacity onPress={() => router.push("/profile-edit")} activeOpacity={0.85} style={{
            backgroundColor: theme.card, borderRadius: 20, padding: 18,
            flexDirection: isRTL ? "row-reverse" : "row", alignItems: "center",
            shadowColor: theme.shadow, shadowOffset: { width: 0, height: 3 }, shadowOpacity: 0.09, elevation: 4,
          }}>
            <View style={{
              width: 48, height: 48, borderRadius: 14,
              backgroundColor: theme.pillBg, alignItems: "center", justifyContent: "center",
              marginRight: isRTL ? 0 : 14, marginLeft: isRTL ? 14 : 0,
            }}>
              <Ionicons name="person-circle-outline" size={26} color={theme.primary} />
            </View>
            <View style={{ flex: 1 }}>
              <Text style={{ fontSize: 15, color: theme.text, fontFamily: "inter-semibold", textAlign: isRTL ? "right" : "left" }}>
                {t("myProfile")}
              </Text>
              <Text style={{ fontSize: 12, color: theme.textSecondary, marginTop: 2, textAlign: isRTL ? "right" : "left" }}>
                {displayName ? t("editInfo") : t("completeProfile")}
              </Text>
            </View>
            {!displayName && (
              <View style={{ backgroundColor: "#E67E22" + "18", borderRadius: 8, paddingHorizontal: 8, paddingVertical: 3, marginRight: 8 }}>
                <Text style={{ fontSize: 10, color: "#E67E22", fontFamily: "inter-semibold" }}>{t("complete")}</Text>
              </View>
            )}
            <Ionicons name={isRTL ? "chevron-back" : "chevron-forward"} size={18} color={theme.textMuted} />
          </TouchableOpacity>
        </View>

        {/* General Section */}
        <View style={{ paddingHorizontal: 20, marginTop: 24 }}>
          <SectionLabel text={t("general")} theme={theme} />
          <View style={card(theme)}>
            <MenuItem icon="settings-outline" label={t("settings")} subtitle={t("langThemeNotif")} onPress={() => router.push("/settings")} />
            <MenuItem icon="mail-outline" label={t("contactUs")} subtitle={t("sendMessage")} onPress={() => router.push("/contact-us")} />
            <MenuItem icon="help-circle-outline" label={t("faq")} subtitle={t("frequentQuestions")} onPress={() => router.push("/faq")} showBorder={false} />
          </View>
        </View>

        {/* App Section */}
        <View style={{ paddingHorizontal: 20, marginTop: 24 }}>
          <SectionLabel text={t("application")} theme={theme} />
          <View style={card(theme)}>
            <MenuItem icon="star-outline" label={t("rateApp")} subtitle={t("shareYourOpinion")} onPress={() => setRateVisible(true)} />
            <MenuItem icon="share-social-outline" label={t("shareApp")} subtitle={t("inviteFriends")} onPress={handleShare} showBorder={false} />
          </View>
        </View>

        {/* Logout */}
        <View style={{ paddingHorizontal: 20, marginTop: 30 }}>
          <TouchableOpacity
            onPress={handleLogout}
            disabled={loggingOut}
            activeOpacity={0.85}
            style={{
              backgroundColor: theme.card, borderRadius: 18,
              paddingVertical: 18, flexDirection: "row",
              alignItems: "center", justifyContent: "center",
              shadowColor: theme.shadow, shadowOffset: { width: 0, height: 2 },
              shadowOpacity: 0.05, elevation: 2, gap: 10,
              opacity: loggingOut ? 0.6 : 1,
            }}
          >
            <Ionicons
              name={loggingOut ? "hourglass-outline" : "log-out-outline"}
              size={22}
              color={theme.danger}
            />
            <Text style={{ fontSize: 15, color: theme.danger, fontFamily: "inter-semibold" }}>
              {loggingOut ? "Déconnexion…" : t("logout")}
            </Text>
          </TouchableOpacity>
        </View>

        <Text style={{ textAlign: "center", fontSize: 12, color: theme.textMuted, marginTop: 20 }}>
          {t("appVersion")}
        </Text>
      </ScrollView>

      <RateModal visible={rateVisible} onClose={() => setRateVisible(false)} />
    </View>
  );
}

import type { Theme } from "@/constants/theme";

function SectionLabel({ text, theme }: { text: string; theme: Theme }) {
  return (
    <Text style={{
      fontSize: 12, color: theme.textSecondary, fontFamily: "inter-semibold",
      textTransform: "uppercase", letterSpacing: 0.8, marginBottom: 10, marginLeft: 4,
    }}>
      {text}
    </Text>
  );
}

function card(theme: Theme) {
  return {
    backgroundColor: theme.card, borderRadius: 18, paddingHorizontal: 16,
    shadowColor: theme.shadow, shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.05, elevation: 2,
  };
}
