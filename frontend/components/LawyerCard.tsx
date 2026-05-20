/**
 * LawyerCard.tsx
 * --------------
 * Reusable, memoized card component for displaying a single lawyer.
 *
 * Props:
 *   - lawyer : Lawyer object from LawyerContext / lawyerService
 *   - theme  : active theme (light / dark)
 */

import React, { memo } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  Linking,
  StyleSheet,
  Alert,
  Platform,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { Lawyer } from "@/services/lawyerService";
import { Theme } from "@/constants/theme";

// ─── Props ────────────────────────────────────────────────────────────────────

interface LawyerCardProps {
  lawyer: Lawyer;
  theme: Theme;
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function openPhone(phone: string, mapsUrl?: string) {
  if (phone) {
    Linking.openURL(`tel:${phone.replace(/\s/g, "")}`);
  } else if (mapsUrl) {
    Linking.openURL(mapsUrl);
  } else {
    Alert.alert("Indisponible", "Le numéro de téléphone n'est pas disponible.");
  }
}

function openWhatsApp(phone: string) {
  if (!phone) {
    Alert.alert("Indisponible", "WhatsApp n'est pas disponible pour ce contact.");
    return;
  }
  const cleaned = phone.replace(/\D/g, "");
  const url =
    Platform.OS === "android"
      ? `whatsapp://send?phone=${cleaned}`
      : `https://wa.me/${cleaned}`;
  Linking.openURL(url).catch(() =>
    Alert.alert("Erreur", "WhatsApp n'est pas installé.")
  );
}

function openAppointment(mapsUrl?: string) {
  if (mapsUrl) {
    Linking.openURL(mapsUrl);
  } else {
    Alert.alert("Rendez-vous", "Fonctionnalité de rendez-vous bientôt disponible.");
  }
}

// ─── Component ────────────────────────────────────────────────────────────────

const LawyerCard = memo(({ lawyer, theme }: LawyerCardProps) => {
  const styles = makeStyles(theme);

  return (
    <View style={styles.card}>
      {/* ── Top Row: Avatar + Info ── */}
      <View style={styles.topRow}>
        {/* Avatar */}
        <View style={styles.avatar}>
          <Text style={styles.avatarText}>{lawyer.avatar}</Text>
        </View>

        {/* Main info */}
        <View style={styles.infoBlock}>
          <Text style={styles.name} numberOfLines={1}>
            {lawyer.name}
          </Text>
          <View style={styles.row}>
            <Ionicons name="briefcase-outline" size={12} color={theme.textSecondary} />
            <Text style={styles.specialty} numberOfLines={1}>
              {" "}{lawyer.specialty}
            </Text>
          </View>
          {lawyer.address ? (
            <View style={styles.row}>
              <Ionicons name="location-outline" size={12} color={theme.textSecondary} />
              <Text style={styles.address} numberOfLines={1}>
                {" "}{lawyer.city || lawyer.address}
              </Text>
            </View>
          ) : null}
        </View>

        {/* Availability badge */}
        <View style={[styles.badge, lawyer.available ? styles.badgeAvailable : styles.badgeBusy]}>
          <Text style={[styles.badgeText, lawyer.available ? styles.badgeTextAvailable : styles.badgeTextBusy]}>
            {lawyer.available ? "Disponible" : "Occupé"}
          </Text>
        </View>
      </View>

      {/* ── Stats Row: experience + rating ── */}
      <View style={styles.statsRow}>
        {lawyer.experience > 0 && (
          <View style={styles.stat}>
            <Ionicons name="time-outline" size={14} color={theme.primary} />
            <Text style={styles.statText}>{lawyer.experience} ans d'exp.</Text>
          </View>
        )}
        {lawyer.rating > 0 && (
          <View style={styles.stat}>
            <Ionicons name="star" size={14} color="#F2C94C" />
            <Text style={styles.statText}>
              {lawyer.rating.toFixed(1)}
              {lawyer.ratingCount > 0 && (
                <Text style={styles.ratingCount}> ({lawyer.ratingCount})</Text>
              )}
            </Text>
          </View>
        )}
        {lawyer.source === "mock" && (
          <View style={styles.stat}>
            <Ionicons name="shield-checkmark-outline" size={14} color={theme.primary} />
            <Text style={styles.statText}>Vérifié</Text>
          </View>
        )}
      </View>

      {/* ── Action Buttons ── */}
      <View style={styles.actions}>
        <TouchableOpacity
          style={[styles.btn, styles.btnPrimary]}
          activeOpacity={0.8}
          onPress={() => openPhone(lawyer.phone, lawyer.mapsUrl)}
          accessibilityLabel={`Appeler ${lawyer.name}`}
        >
          <Ionicons name="call-outline" size={16} color="#fff" />
          <Text style={styles.btnTextPrimary}>Appeler</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.btn, styles.btnWhatsApp]}
          activeOpacity={0.8}
          onPress={() => openWhatsApp(lawyer.phone)}
          accessibilityLabel={`WhatsApp ${lawyer.name}`}
        >
          <Ionicons name="logo-whatsapp" size={16} color="#fff" />
          <Text style={styles.btnTextPrimary}>WhatsApp</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.btn, styles.btnOutline]}
          activeOpacity={0.8}
          onPress={() => openAppointment(lawyer.mapsUrl)}
          accessibilityLabel={`Rendez-vous avec ${lawyer.name}`}
        >
          <Ionicons name="calendar-outline" size={16} color={theme.primary} />
          <Text style={[styles.btnTextOutline, { color: theme.primary }]}>RDV</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
});

LawyerCard.displayName = "LawyerCard";
export default LawyerCard;

// ─── Styles ───────────────────────────────────────────────────────────────────

function makeStyles(theme: Theme) {
  return StyleSheet.create({
    card: {
      backgroundColor: theme.card,
      borderRadius: 20,
      padding: 16,
      marginHorizontal: 16,
      marginVertical: 8,
      shadowColor: theme.shadow,
      shadowOffset: { width: 0, height: 3 },
      shadowOpacity: 0.1,
      shadowRadius: 8,
      elevation: 4,
      borderWidth: 1,
      borderColor: theme.border,
    },
    topRow: {
      flexDirection: "row",
      alignItems: "flex-start",
      marginBottom: 12,
    },
    avatar: {
      width: 52,
      height: 52,
      borderRadius: 26,
      backgroundColor: theme.primaryLight,
      alignItems: "center",
      justifyContent: "center",
      marginRight: 12,
      borderWidth: 2,
      borderColor: theme.primaryMedium,
    },
    avatarText: {
      fontSize: 16,
      fontFamily: "inter-semibold",
      color: theme.primary,
    },
    infoBlock: {
      flex: 1,
      gap: 3,
    },
    name: {
      fontSize: 15,
      fontFamily: "inter-semibold",
      color: theme.text,
    },
    row: {
      flexDirection: "row",
      alignItems: "center",
    },
    specialty: {
      fontSize: 12,
      color: theme.textSecondary,
      flexShrink: 1,
    },
    address: {
      fontSize: 12,
      color: theme.textMuted,
      flexShrink: 1,
    },
    badge: {
      borderRadius: 12,
      paddingHorizontal: 10,
      paddingVertical: 4,
      alignSelf: "flex-start",
      marginLeft: 8,
    },
    badgeAvailable: {
      backgroundColor: "rgba(39,174,96,0.12)",
    },
    badgeBusy: {
      backgroundColor: "rgba(224,69,69,0.12)",
    },
    badgeText: {
      fontSize: 11,
      fontFamily: "inter-semibold",
    },
    badgeTextAvailable: {
      color: "#27AE60",
    },
    badgeTextBusy: {
      color: "#E04545",
    },
    statsRow: {
      flexDirection: "row",
      gap: 16,
      marginBottom: 14,
      flexWrap: "wrap",
    },
    stat: {
      flexDirection: "row",
      alignItems: "center",
      gap: 4,
    },
    statText: {
      fontSize: 12,
      fontFamily: "inter-medium",
      color: theme.textSecondary,
    },
    ratingCount: {
      fontSize: 11,
      color: theme.textMuted,
    },
    actions: {
      flexDirection: "row",
      gap: 8,
    },
    btn: {
      flex: 1,
      flexDirection: "row",
      alignItems: "center",
      justifyContent: "center",
      paddingVertical: 10,
      borderRadius: 12,
      gap: 5,
    },
    btnPrimary: {
      backgroundColor: theme.primary,
    },
    btnWhatsApp: {
      backgroundColor: "#25D366",
    },
    btnOutline: {
      borderWidth: 1.5,
      borderColor: theme.primary,
      backgroundColor: "transparent",
    },
    btnTextPrimary: {
      fontSize: 12,
      fontFamily: "inter-semibold",
      color: "#fff",
    },
    btnTextOutline: {
      fontSize: 12,
      fontFamily: "inter-semibold",
    },
  });
}
