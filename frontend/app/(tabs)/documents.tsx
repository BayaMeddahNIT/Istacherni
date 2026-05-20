import { useState, useEffect, useMemo, useCallback, useRef } from "react";
import {
  Text, View, ScrollView, TouchableOpacity, FlatList,
  TextInput, Linking, Alert, Platform, ActivityIndicator,
  Modal, Animated, Pressable,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import * as Location from "expo-location";
import { useTheme, useTranslation } from "@/context/UserContext";
import { useLocalSearchParams } from "expo-router";
import { fetchLawyerPhone, fetchLawyerDetails, type Lawyer as LawyerType } from "@/services/lawyerService";
import { apiFetch } from "@/services/api";
import { useLawyerContext } from "@/context/LawyerContext";
import { activityService } from "@/services/activityService";

// ─── Types ────────────────────────────────────────────────────────────────────

type Screen = "hub" | "notes" | "lawyers";



// ─── Law Categories ───────────────────────────────────────────────────────────

const LAW_CATEGORIES = [
  {
    id: "labor",
    titleKey: "catLaborLaw" as const,
    descKey: "catLaborDesc" as const,
    icon: "briefcase-outline",
    ref: "Loi n° 90-11 du 21 avril 1990",
    url: "https://app.univ-blida2.dz/cours/documents/pdf427.pdf",
    accent: "#C4885C",
  },
  {
    id: "commercial",
    titleKey: "catCommercialLaw" as const,
    descKey: "catCommercialDesc" as const,
    icon: "trending-up-outline",
    ref: "Ordonnance n° 75-59 du 26 septembre 1975",
    url: "https://publications.univ-blida2.dz/documents/pdf373.pdf",
    accent: "#5B82A8",
  },
  {
    id: "civil",
    titleKey: "catCivilLaw" as const,
    descKey: "catCivilDesc" as const,
    icon: "home-outline",
    ref: "Ordonnance n° 75-58 du 26 septembre 1975",
    url: "https://menarights.org/sites/default/files/2016-12/ALG_Codecivil2007_AR.pdf",
    accent: "#5B967A",
  },
  {
    id: "criminal",
    titleKey: "catCriminalLaw" as const,
    descKey: "catCriminalDesc" as const,
    icon: "shield-checkmark-outline",
    ref: "Ordonnance n° 66-156 du 8 juin 1966",
    url: "https://www.vertic.org/media/National%20Legislation/Algeria/DZ_Code_Penal.pdf",
    accent: "#9B6B6B",
  },
];


// ─── Category Card ────────────────────────────────────────────────────────────


function CategoryCard({
  cat, theme, t, isRTL, isLoading, onPress,
}: {
  cat: typeof LAW_CATEGORIES[0];
  theme: any; t: any; isRTL: boolean;
  isLoading: boolean; onPress: () => void;
}) {
  return (
    <View style={{
      backgroundColor: theme.card, borderRadius: 20, overflow: "hidden",
      shadowColor: theme.shadow, shadowOffset: { width: 0, height: 3 },
      shadowOpacity: 0.09, elevation: 4,
      borderWidth: 1, borderColor: theme.border,
    }}>
      {/* Coloured top accent strip */}
      <View style={{ height: 4, backgroundColor: cat.accent }} />

      <View style={{ padding: 14 }}>
        {/* Icon */}
        <View style={{
          width: 46, height: 46, borderRadius: 14,
          backgroundColor: cat.accent + "18",
          alignItems: "center", justifyContent: "center",
          borderWidth: 1.5, borderColor: cat.accent + "35",
          marginBottom: 10,
          alignSelf: isRTL ? "flex-end" : "flex-start",
        }}>
          <Ionicons name={cat.icon as any} size={22} color={cat.accent} />
        </View>

        {/* Title */}
        <Text style={{ fontSize: 14, fontFamily: "inter-semibold", color: theme.text, marginBottom: 5, textAlign: isRTL ? "right" : "left" }}>
          {t(cat.titleKey)}
        </Text>

        {/* Description */}
        <Text style={{ fontSize: 11, color: theme.textSecondary, lineHeight: 16, marginBottom: 10, textAlign: isRTL ? "right" : "left" }} numberOfLines={3}>
          {t(cat.descKey)}
        </Text>

        {/* Reference pill */}
        <View style={{ backgroundColor: theme.pillBg, borderRadius: 8, paddingHorizontal: 8, paddingVertical: 4, marginBottom: 12, alignSelf: isRTL ? "flex-end" : "flex-start" }}>
          <Text style={{ fontSize: 9, color: theme.textMuted, fontFamily: "inter-medium" }} numberOfLines={1}>
            {cat.ref}
          </Text>
        </View>

        {/* Open PDF button */}
        <TouchableOpacity
          onPress={onPress}
          activeOpacity={0.75}
          disabled={isLoading}
          style={{
            backgroundColor: cat.accent, borderRadius: 12, paddingVertical: 9,
            flexDirection: isRTL ? "row-reverse" : "row",
            alignItems: "center", justifyContent: "center", gap: 6,
            opacity: isLoading ? 0.7 : 1,
          }}
        >
          <Ionicons name={isLoading ? "hourglass-outline" : "document-text-outline"} size={14} color="#fff" />
          <Text style={{ fontSize: 12, fontFamily: "inter-semibold", color: "#fff" }}>
            {isLoading ? "…" : t("openPdf")}
          </Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

// ─── Sub-screen: Legal Notes (Category Hub) ───────────────────────────────────

function NotesScreen({ onBack, resumeDocId }: { onBack: () => void; resumeDocId?: string }) {
  const theme = useTheme();
  const { t, isRTL } = useTranslation();
  const [loading, setLoading] = useState<string | null>(null);
  const hasResumed = useRef(false);

  const openPdf = async (cat: typeof LAW_CATEGORIES[0]) => {
    setLoading(cat.id);
    try {
      // Log activity
      await activityService.addActivity({
        type: "document",
        title: t(cat.titleKey),
        category: t("legalMemoirs"),
        documentId: cat.id,
      });

      const canOpen = await Linking.canOpenURL(cat.url);
      if (canOpen) {
        await Linking.openURL(cat.url);
      } else {
        Alert.alert("Erreur", "Impossible d'ouvrir ce document PDF.");
      }
    } catch {
      Alert.alert("Erreur", "Une erreur est survenue.");
    } finally {
      setLoading(null);
    }
  };

  useEffect(() => {
    if (resumeDocId && !hasResumed.current) {
      const doc = LAW_CATEGORIES.find(c => c.id === resumeDocId);
      if (doc) {
        hasResumed.current = true;
        openPdf(doc);
      }
    }
  }, [resumeDocId]);

  return (
    <View style={{ flex: 1, backgroundColor: theme.background }}>

      {/* Header */}
      <View style={{
        paddingTop: Platform.OS === "ios" ? 56 : 44,
        paddingHorizontal: 16, paddingBottom: 20,
        backgroundColor: theme.headerBg,
        borderBottomLeftRadius: 28, borderBottomRightRadius: 28,
      }}>
        <View style={{ flexDirection: isRTL ? "row-reverse" : "row", alignItems: "center", gap: 12, marginBottom: 12 }}>
          <TouchableOpacity
            onPress={onBack}
            style={{ width: 38, height: 38, borderRadius: 12, backgroundColor: theme.primaryMedium, alignItems: "center", justifyContent: "center" }}
          >
            <Ionicons name={isRTL ? "arrow-forward" : "arrow-back"} size={20} color={theme.primary} />
          </TouchableOpacity>
          <View style={{ flex: 1 }}>
            <Text style={{ fontSize: 18, fontFamily: "inter-semibold", color: theme.text, textAlign: isRTL ? "right" : "left" }}>
              {t("legalMemoirs")}
            </Text>
            <Text style={{ fontSize: 12, color: theme.textSecondary, marginTop: 2, textAlign: isRTL ? "right" : "left" }}>
              {t("chooseLawCategorySub")}
            </Text>
          </View>
        </View>

        {/* Count pill */}
        <View style={{ flexDirection: isRTL ? "row-reverse" : "row" }}>
          <View style={{ backgroundColor: theme.primaryLight, borderRadius: 20, paddingHorizontal: 12, paddingVertical: 5, flexDirection: "row", alignItems: "center", gap: 6 }}>
            <Ionicons name="library-outline" size={13} color={theme.primary} />
            <Text style={{ fontSize: 12, fontFamily: "inter-semibold", color: theme.primary }}>
              {LAW_CATEGORIES.length} {t("chooseLawCategory")}
            </Text>
          </View>
        </View>
      </View>

      {/* Grid */}
      <ScrollView contentContainerStyle={{ padding: 16, gap: 14 }} showsVerticalScrollIndicator={false}>
        <View style={{ flexDirection: isRTL ? "row-reverse" : "row", gap: 14 }}>

          {/* Column 1: Labor + Civil */}
          <View style={{ flex: 1, gap: 14 }}>
            {[LAW_CATEGORIES[0], LAW_CATEGORIES[2]].map(cat => (
              <CategoryCard
                key={cat.id} cat={cat} theme={theme} t={t} isRTL={isRTL}
                isLoading={loading === cat.id} onPress={() => openPdf(cat)}
              />
            ))}
          </View>

          {/* Column 2: Commercial + Criminal */}
          <View style={{ flex: 1, gap: 14 }}>
            {[LAW_CATEGORIES[1], LAW_CATEGORIES[3]].map(cat => (
              <CategoryCard
                key={cat.id} cat={cat} theme={theme} t={t} isRTL={isRTL}
                isLoading={loading === cat.id} onPress={() => openPdf(cat)}
              />
            ))}
          </View>
        </View>

        {/* Info note */}
        <View style={{ backgroundColor: theme.card, borderRadius: 16, padding: 14, flexDirection: isRTL ? "row-reverse" : "row", gap: 10, alignItems: "flex-start", borderWidth: 1, borderColor: theme.border }}>
          <View style={{ width: 32, height: 32, borderRadius: 10, backgroundColor: theme.primaryLight, alignItems: "center", justifyContent: "center", marginTop: 1 }}>
            <Ionicons name="information-circle-outline" size={17} color={theme.primary} />
          </View>
          <Text style={{ flex: 1, fontSize: 12, color: theme.textSecondary, lineHeight: 19, textAlign: isRTL ? "right" : "left" }}>
            {t("pdfSource")} — Université Blida 2, MenaRights &amp; VERTIC
          </Text>
        </View>
      </ScrollView>
    </View>
  );
}

// ─── Sub-screen: Find a Lawyer ────────────────────────────────────────────────

// Sub-screen: Find a Lawyer

function LawyersScreen({ onBack }: { onBack: () => void }) {
  const theme = useTheme();
  const { t, isRTL } = useTranslation();
  const [query, setQuery] = useState("");
  const [sheetLawyer, setSheetLawyer] = useState<LawyerType | null>(null);
  const [contactLawyer, setContactLawyer] = useState<LawyerType | null>(null);
  const [fetchedPhone, setFetchedPhone] = useState<string | null>(null);
  const [phoneFetching, setPhoneFetching] = useState(false);
  const sheetAnim = useRef(new Animated.Value(0)).current;

  // ── Read from shared LawyerContext (populated by map.tsx) ──────────────────
  const { lawyers, loading: isLoading, userCoords } = useLawyerContext();

  const filtered = useMemo(
    () => lawyers.filter((l: LawyerType) =>
      l.name.toLowerCase().includes(query.toLowerCase()) ||
      l.specialty.toLowerCase().includes(query.toLowerCase()) ||
      l.city.toLowerCase().includes(query.toLowerCase())
    ),
    [lawyers, query]
  );

  const openSheet = useCallback((lawyer: LawyerType) => {
    setSheetLawyer(lawyer);
    setFetchedPhone(null);
    Animated.spring(sheetAnim, { toValue: 1, useNativeDriver: true, tension: 80, friction: 14 }).start();
    // Pre-fetch phone for Google lawyers
    if (lawyer.source === "google" && !lawyer.phone) {
      setPhoneFetching(true);
      fetchLawyerDetails(lawyer.id).then(d => {
        setFetchedPhone(d.phone);
        setPhoneFetching(false);
      });
    }
  }, [sheetAnim]);

  const closeSheet = useCallback(() => {
    Animated.timing(sheetAnim, { toValue: 0, duration: 220, useNativeDriver: true }).start(() => setSheetLawyer(null));
  }, [sheetAnim]);

  const openContact = useCallback((lawyer: LawyerType) => {
    setContactLawyer(lawyer);
    if (lawyer.source === "google" && !lawyer.phone && !fetchedPhone) {
      setPhoneFetching(true);
      fetchLawyerDetails(lawyer.id).then(d => {
        setFetchedPhone(d.phone);
        setPhoneFetching(false);
      });
    }
  }, [fetchedPhone]);

  const callLawyer = useCallback(async (lawyer: LawyerType) => {
    const phone = lawyer.phone || fetchedPhone;
    if (phone) { Linking.openURL("tel:" + phone.replace(/\s/g, "")); }
    else if (lawyer.mapsUrl) { Linking.openURL(lawyer.mapsUrl); }
  }, [fetchedPhone]);

  const openNavigation = useCallback((lawyer: LawyerType) => {
    if (!lawyer.latitude || !lawyer.longitude) {
      if (lawyer.mapsUrl) { Linking.openURL(lawyer.mapsUrl); return; }
      Alert.alert("Navigation", "Coordonnées non disponibles pour cet avocat.");
      return;
    }
    const label = encodeURIComponent(lawyer.name);
    const url = Platform.OS === "ios"
      ? `maps://app?daddr=${lawyer.latitude},${lawyer.longitude}&q=${label}`
      : `https://www.google.com/maps/dir/?api=1&destination=${lawyer.latitude},${lawyer.longitude}`;
    Linking.openURL(url).catch(() =>
      Linking.openURL(`https://www.google.com/maps/dir/?api=1&destination=${lawyer.latitude},${lawyer.longitude}`)
    );
  }, []);

  const distKm = useCallback((lawyer: LawyerType): number | null => {
    if (!userCoords || !lawyer.latitude || !lawyer.longitude) return null;
    const R = 6371;
    const dLat = ((lawyer.latitude - userCoords.latitude) * Math.PI) / 180;
    const dLon = ((lawyer.longitude - userCoords.longitude) * Math.PI) / 180;
    const a = Math.sin(dLat/2)**2 + Math.cos(userCoords.latitude*Math.PI/180)*Math.cos(lawyer.latitude*Math.PI/180)*Math.sin(dLon/2)**2;
    return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
  }, [userCoords]);

  const fmtDist = (km: number) => km < 1 ? `${Math.round(km*1000)} m` : `${km.toFixed(1)} km`;
  const fmtTime = (km: number) => { const m = Math.max(1,Math.round((km/25)*60)); return m<60?`${m} min`:`${Math.floor(m/60)}h${m%60}min`; };


  return (
    <View style={{ flex: 1, backgroundColor: theme.background }}>
      <View style={{ paddingTop: Platform.OS === "ios" ? 56 : 44, paddingHorizontal: 16, paddingBottom: 16, backgroundColor: theme.headerBg, borderBottomLeftRadius: 24, borderBottomRightRadius: 24 }}>
        <View style={{ flexDirection: isRTL ? "row-reverse" : "row", alignItems: "center", gap: 12, marginBottom: 16 }}>
          <TouchableOpacity onPress={onBack} style={{ width: 38, height: 38, borderRadius: 12, backgroundColor: theme.primaryMedium, alignItems: "center", justifyContent: "center" }}>
            <Ionicons name={isRTL ? "arrow-forward" : "arrow-back"} size={20} color={theme.primary} />
          </TouchableOpacity>
          <View style={{ flex: 1 }}>
            <Text style={{ fontSize: 18, fontFamily: "inter-semibold", color: theme.text, textAlign: isRTL ? "right" : "left" }}>{t("lawyersAvailable")}</Text>
            <Text style={{ fontSize: 12, color: theme.textSecondary, marginTop: 1, textAlign: isRTL ? "right" : "left" }}>
              {isLoading ? t("loadingLawyers") : filtered.length + " " + t("lawyersReferenced")}
            </Text>
          </View>
        </View>
        <View style={{ flexDirection: isRTL ? "row-reverse" : "row", alignItems: "center", backgroundColor: theme.card, borderRadius: 14, paddingHorizontal: 14, height: 44, gap: 8 }}>
          <Ionicons name="search" size={18} color={theme.textMuted} />
          <TextInput placeholder={t("searchLawyerPlaceholder")} placeholderTextColor={theme.textMuted} value={query} onChangeText={setQuery} style={{ flex: 1, fontSize: 14, color: theme.text, textAlign: isRTL ? "right" : "left" }} />
          {query.length > 0 && (<TouchableOpacity onPress={() => setQuery("")}><Ionicons name="close-circle" size={18} color={theme.textMuted} /></TouchableOpacity>)}
        </View>
      </View>


      {isLoading && (
        <View style={{ flex: 1, alignItems: "center", justifyContent: "center", gap: 16 }}>
          <ActivityIndicator size="large" color={theme.primary} />
          <Text style={{ fontSize: 14, color: theme.textSecondary }}>{t("loadingLawyers")}</Text>
        </View>
      )}

      {!isLoading && (
        <FlatList
          data={filtered}
          keyExtractor={l => l.id}
          contentContainerStyle={{ padding: 16, gap: 12, paddingBottom: 32 }}
          showsVerticalScrollIndicator={false}
          ListEmptyComponent={
            <View style={{ alignItems: "center", paddingTop: 60 }}>
              <Ionicons name="person-outline" size={48} color={theme.textMuted} />
              <Text style={{ fontSize: 14, color: theme.textMuted, marginTop: 12 }}>{t("noLawyerFound")}</Text>
            </View>
          }
          renderItem={({ item }) => (
            <View style={{ backgroundColor: theme.card, borderRadius: 20, padding: 16, shadowColor: theme.shadow, shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.07, elevation: 3, borderWidth: 1, borderColor: theme.border }}>
              <View style={{ flexDirection: isRTL ? "row-reverse" : "row", gap: 14, alignItems: "flex-start" }}>
                <View style={{ width: 56, height: 56, borderRadius: 18, backgroundColor: theme.primaryLight, borderWidth: 2, borderColor: theme.primaryMedium, alignItems: "center", justifyContent: "center" }}>
                  <Text style={{ fontSize: 16, fontFamily: "inter-semibold", color: theme.primary }}>{item.avatar}</Text>
                </View>
                <View style={{ flex: 1 }}>
                  <View style={{ flexDirection: isRTL ? "row-reverse" : "row", alignItems: "center", justifyContent: "space-between" }}>
                    <Text style={{ fontSize: 15, fontFamily: "inter-semibold", color: theme.text, flex: 1, textAlign: isRTL ? "right" : "left" }} numberOfLines={1}>{item.name}</Text>
                    <View style={{ backgroundColor: item.available ? theme.success + "18" : theme.pillBg, borderRadius: 8, paddingHorizontal: 8, paddingVertical: 3 }}>
                      <Text style={{ fontSize: 10, fontFamily: "inter-semibold", color: item.available ? theme.success : theme.textMuted }}>{item.available ? t("available") : t("busy")}</Text>
                    </View>
                  </View>
                  <View style={{ backgroundColor: theme.primaryLight, borderRadius: 8, paddingHorizontal: 8, paddingVertical: 3, alignSelf: isRTL ? "flex-end" : "flex-start", marginTop: 4 }}>
                    <Text style={{ fontSize: 11, fontFamily: "inter-semibold", color: theme.primary }}>{item.specialty}</Text>
                  </View>
                  {/* Bio description */}
                  {!!item.bio && (
                    <Text style={{ fontSize: 12, color: theme.textSecondary, lineHeight: 17, marginTop: 6, marginBottom: 2, textAlign: isRTL ? "right" : "left" }} numberOfLines={2}>{item.bio}</Text>
                  )}
                  <View style={{ marginTop: 8, gap: 3 }}>
                    {!!item.address && (
                      <View style={{ flexDirection: isRTL ? "row-reverse" : "row", alignItems: "center", gap: 6 }}>
                        <Ionicons name="location-outline" size={12} color={theme.textMuted} />
                        <Text style={{ fontSize: 12, color: theme.textSecondary, flex: 1 }} numberOfLines={1}>{item.city && item.address !== item.city ? item.address + ", " + item.city : item.address}</Text>
                      </View>
                    )}
                    {item.experience > 0 && (
                      <View style={{ flexDirection: isRTL ? "row-reverse" : "row", alignItems: "center", gap: 6 }}>
                        <Ionicons name="briefcase-outline" size={12} color={theme.textMuted} />
                        <Text style={{ fontSize: 12, color: theme.textSecondary }}>{item.experience} {t("experience")}</Text>
                      </View>
                    )}
                    {item.rating > 0 && (
                      <View style={{ flexDirection: isRTL ? "row-reverse" : "row", alignItems: "center", gap: 3 }}>
                        {[1,2,3,4,5].map(s => (<Ionicons key={s} name={s <= Math.floor(item.rating) ? "star" : "star-outline"} size={12} color="#F2C94C" />))}
                        <Text style={{ fontSize: 11, color: theme.textMuted, marginLeft: 4 }}>{item.rating.toFixed(1)}{item.ratingCount > 0 ? " (" + item.ratingCount + ")" : ""}</Text>
                      </View>
                    )}
                  </View>
                </View>
              </View>
              {/* ── Contact info row ── */}
              {(item.phone || item.email || (userCoords && item.latitude)) && (
                <View style={{ flexDirection: isRTL ? "row-reverse" : "row", flexWrap: "wrap", gap: 6, marginTop: 10, paddingTop: 10, borderTopWidth: 1, borderTopColor: theme.border }}>
                  {!!item.phone && (
                    <TouchableOpacity onPress={() => Linking.openURL("tel:" + item.phone.replace(/\s/g, ""))} style={{ flexDirection: "row", alignItems: "center", gap: 4, backgroundColor: theme.pillBg, borderRadius: 8, paddingHorizontal: 8, paddingVertical: 4 }}>
                      <Ionicons name="call-outline" size={12} color={theme.primary} />
                      <Text style={{ fontSize: 11, color: theme.primary, fontFamily: "inter-medium" }}>{item.phone}</Text>
                    </TouchableOpacity>
                  )}
                  {!!item.email && (
                    <TouchableOpacity onPress={() => Linking.openURL("mailto:" + item.email)} style={{ flexDirection: "row", alignItems: "center", gap: 4, backgroundColor: theme.pillBg, borderRadius: 8, paddingHorizontal: 8, paddingVertical: 4 }}>
                      <Ionicons name="mail-outline" size={12} color={theme.primary} />
                      <Text style={{ fontSize: 11, color: theme.primary, fontFamily: "inter-medium" }} numberOfLines={1}>{item.email}</Text>
                    </TouchableOpacity>
                  )}
                  {userCoords && item.latitude && item.longitude && (() => {
                    const R = 6371;
                    const dLat = ((item.latitude - userCoords.latitude) * Math.PI) / 180;
                    const dLon = ((item.longitude - userCoords.longitude) * Math.PI) / 180;
                    const a = Math.sin(dLat/2)**2 + Math.cos(userCoords.latitude*Math.PI/180)*Math.cos(item.latitude*Math.PI/180)*Math.sin(dLon/2)**2;
                    const km = R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
                    const distStr = km < 1 ? `${Math.round(km*1000)} m` : `${km.toFixed(1)} km`;
                    const minDrive = Math.max(1, Math.round((km/25)*60));
                    return (
                      <View style={{ flexDirection: "row", alignItems: "center", gap: 4, backgroundColor: theme.primaryLight, borderRadius: 8, paddingHorizontal: 8, paddingVertical: 4 }}>
                        <Ionicons name="navigate-outline" size={12} color={theme.primary} />
                        <Text style={{ fontSize: 11, color: theme.primary, fontFamily: "inter-medium" }}>{distStr} · {minDrive} min</Text>
                      </View>
                    );
                  })()}
                </View>
              )}
              <View style={{ flexDirection: "row", gap: 8, marginTop: 14 }}>
                <TouchableOpacity
                  onPress={() => openContact(item)}
                  style={{ flex: 1, flexDirection: "row", alignItems: "center", justifyContent: "center", gap: 6, backgroundColor: theme.success + "18", borderRadius: 12, paddingVertical: 11, borderWidth: 1, borderColor: theme.success + "35" }}
                >
                  <Ionicons name="chatbubble-ellipses-outline" size={15} color={theme.success} />
                  <Text style={{ fontSize: 12, fontFamily: "inter-semibold", color: theme.success }}>Contacter</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  onPress={() => openSheet(item)}
                  style={{ flex: 1, flexDirection: "row", alignItems: "center", justifyContent: "center", gap: 6, backgroundColor: theme.primaryLight, borderRadius: 12, paddingVertical: 11, borderWidth: 1, borderColor: theme.primaryMedium }}
                >
                  <Ionicons name="information-circle-outline" size={15} color={theme.primary} />
                  <Text style={{ fontSize: 12, fontFamily: "inter-semibold", color: theme.primary }}>Détails</Text>
                </TouchableOpacity>
              </View>
            </View>
          )}
        />
      )}

      {/* ── Contact Options Modal ── */}
      <Modal visible={!!contactLawyer} transparent animationType="fade" onRequestClose={() => setContactLawyer(null)}>
        <Pressable style={{ flex: 1, backgroundColor: "rgba(0,0,0,0.55)", justifyContent: "flex-end" }} onPress={() => setContactLawyer(null)}>
          <Pressable style={{ backgroundColor: theme.card, borderTopLeftRadius: 28, borderTopRightRadius: 28, padding: 24, paddingBottom: Platform.OS === "ios" ? 40 : 24 }}>
            <View style={{ width: 40, height: 4, backgroundColor: theme.border, borderRadius: 2, alignSelf: "center", marginBottom: 20 }} />
            <Text style={{ fontSize: 16, fontFamily: "inter-semibold", color: theme.text, marginBottom: 4 }} numberOfLines={1}>{contactLawyer?.name}</Text>
            <Text style={{ fontSize: 12, color: theme.textSecondary, marginBottom: 20 }}>{contactLawyer?.specialty}</Text>
            {phoneFetching && <ActivityIndicator color={theme.primary} style={{ marginBottom: 16 }} />}
            {/* Call */}
            {(contactLawyer?.phone || fetchedPhone) && (
              <TouchableOpacity onPress={() => { callLawyer(contactLawyer!); setContactLawyer(null); }} style={{ flexDirection: "row", alignItems: "center", gap: 14, backgroundColor: theme.success+"14", borderRadius: 16, padding: 16, marginBottom: 10, borderWidth: 1, borderColor: theme.success+"30" }}>
                <View style={{ width: 44, height: 44, borderRadius: 14, backgroundColor: theme.success+"22", alignItems: "center", justifyContent: "center" }}>
                  <Ionicons name="call" size={22} color={theme.success} />
                </View>
                <View>
                  <Text style={{ fontSize: 14, fontFamily: "inter-semibold", color: theme.text }}>Appeler</Text>
                  <Text style={{ fontSize: 12, color: theme.textSecondary }}>{contactLawyer?.phone || fetchedPhone}</Text>
                </View>
              </TouchableOpacity>
            )}
            {/* WhatsApp */}
            {(contactLawyer?.phone || fetchedPhone) && (
              <TouchableOpacity onPress={() => { Linking.openURL("https://wa.me/" + (contactLawyer?.phone||fetchedPhone||'').replace(/[^0-9]/g,"")); setContactLawyer(null); }} style={{ flexDirection: "row", alignItems: "center", gap: 14, backgroundColor: "#25D36614", borderRadius: 16, padding: 16, marginBottom: 10, borderWidth: 1, borderColor: "#25D36630" }}>
                <View style={{ width: 44, height: 44, borderRadius: 14, backgroundColor: "#25D36622", alignItems: "center", justifyContent: "center" }}>
                  <Ionicons name="logo-whatsapp" size={22} color="#25D366" />
                </View>
                <Text style={{ fontSize: 14, fontFamily: "inter-semibold", color: theme.text }}>WhatsApp</Text>
              </TouchableOpacity>
            )}
            {/* Email */}
            {!!contactLawyer?.email && (
              <TouchableOpacity onPress={() => { Linking.openURL("mailto:"+contactLawyer.email!); setContactLawyer(null); }} style={{ flexDirection: "row", alignItems: "center", gap: 14, backgroundColor: "#4285F414", borderRadius: 16, padding: 16, marginBottom: 10, borderWidth: 1, borderColor: "#4285F430" }}>
                <View style={{ width: 44, height: 44, borderRadius: 14, backgroundColor: "#4285F422", alignItems: "center", justifyContent: "center" }}>
                  <Ionicons name="mail" size={22} color="#4285F4" />
                </View>
                <View>
                  <Text style={{ fontSize: 14, fontFamily: "inter-semibold", color: theme.text }}>Email</Text>
                  <Text style={{ fontSize: 12, color: theme.textSecondary }}>{contactLawyer?.email}</Text>
                </View>
              </TouchableOpacity>
            )}
            {!contactLawyer?.phone && !fetchedPhone && !contactLawyer?.email && !phoneFetching && (
              <Text style={{ color: theme.textMuted, textAlign: "center", paddingVertical: 12 }}>Aucune coordonnée disponible</Text>
            )}
          </Pressable>
        </Pressable>
      </Modal>

      {/* ── Lawyer Detail Bottom Sheet ── */}
      <Modal visible={!!sheetLawyer} transparent animationType="none" onRequestClose={closeSheet}>
        <Pressable style={{ flex: 1, backgroundColor: "rgba(0,0,0,0.5)" }} onPress={closeSheet}>
          <Animated.View
            style={[{ position: "absolute", bottom: 0, left: 0, right: 0, backgroundColor: theme.card, borderTopLeftRadius: 28, borderTopRightRadius: 28, padding: 24, paddingBottom: Platform.OS === "ios" ? 44 : 28 },
              { transform: [{ translateY: sheetAnim.interpolate({ inputRange: [0,1], outputRange: [400, 0] }) }] }]}
          >
            <Pressable>
              <View style={{ width: 40, height: 4, backgroundColor: theme.border, borderRadius: 2, alignSelf: "center", marginBottom: 20 }} />
              {/* Avatar + name */}
              <View style={{ flexDirection: "row", alignItems: "center", gap: 14, marginBottom: 16 }}>
                <View style={{ width: 58, height: 58, borderRadius: 18, backgroundColor: theme.primaryLight, borderWidth: 2, borderColor: theme.primaryMedium, alignItems: "center", justifyContent: "center" }}>
                  <Text style={{ fontSize: 18, fontFamily: "inter-semibold", color: theme.primary }}>{sheetLawyer?.avatar}</Text>
                </View>
                <View style={{ flex: 1 }}>
                  <Text style={{ fontSize: 16, fontFamily: "inter-semibold", color: theme.text, marginBottom: 4 }}>{sheetLawyer?.name}</Text>
                  <View style={{ flexDirection: "row", gap: 6, flexWrap: "wrap" }}>
                    <View style={{ backgroundColor: theme.primaryLight, borderRadius: 8, paddingHorizontal: 8, paddingVertical: 3 }}>
                      <Text style={{ fontSize: 11, fontFamily: "inter-semibold", color: theme.primary }}>{sheetLawyer?.specialty}</Text>
                    </View>
                    <View style={{ backgroundColor: sheetLawyer?.available ? theme.success+"18" : theme.pillBg, borderRadius: 8, paddingHorizontal: 8, paddingVertical: 3 }}>
                      <Text style={{ fontSize: 11, fontFamily: "inter-semibold", color: sheetLawyer?.available ? theme.success : theme.textMuted }}>{sheetLawyer?.available ? "Disponible" : "Occupé"}</Text>
                    </View>
                  </View>
                </View>
              </View>
              {/* Stats row */}
              <View style={{ flexDirection: "row", gap: 8, marginBottom: 16 }}>
                {sheetLawyer?.rating! > 0 && (
                  <View style={{ flex: 1, backgroundColor: theme.pillBg, borderRadius: 14, padding: 12, alignItems: "center", gap: 4 }}>
                    <Ionicons name="star" size={18} color="#F2C94C" />
                    <Text style={{ fontSize: 14, fontFamily: "inter-semibold", color: theme.text }}>{sheetLawyer?.rating.toFixed(1)}</Text>
                    <Text style={{ fontSize: 10, color: theme.textMuted }}>Note</Text>
                  </View>
                )}
                {(() => { const km = sheetLawyer ? distKm(sheetLawyer) : null; return km !== null ? (
                  <View style={{ flex: 1, backgroundColor: theme.primaryLight, borderRadius: 14, padding: 12, alignItems: "center", gap: 4 }}>
                    <Ionicons name="navigate" size={18} color={theme.primary} />
                    <Text style={{ fontSize: 14, fontFamily: "inter-semibold", color: theme.primary }}>{fmtDist(km)}</Text>
                    <Text style={{ fontSize: 10, color: theme.primary+"99" }}>{fmtTime(km)}</Text>
                  </View>
                ) : null; })()}
                {sheetLawyer?.experience! > 0 && (
                  <View style={{ flex: 1, backgroundColor: theme.pillBg, borderRadius: 14, padding: 12, alignItems: "center", gap: 4 }}>
                    <Ionicons name="briefcase" size={18} color={theme.textSecondary} />
                    <Text style={{ fontSize: 14, fontFamily: "inter-semibold", color: theme.text }}>{sheetLawyer?.experience} ans</Text>
                    <Text style={{ fontSize: 10, color: theme.textMuted }}>Expérience</Text>
                  </View>
                )}
              </View>
              {/* Address */}
              {!!sheetLawyer?.address && (
                <View style={{ flexDirection: "row", alignItems: "center", gap: 8, marginBottom: 20, backgroundColor: theme.pillBg, borderRadius: 12, padding: 12 }}>
                  <Ionicons name="location-outline" size={16} color={theme.textMuted} />
                  <Text style={{ fontSize: 13, color: theme.textSecondary, flex: 1 }}>{sheetLawyer?.address}</Text>
                </View>
              )}
              {/* Action buttons */}
              <View style={{ flexDirection: "row", gap: 10 }}>
                <TouchableOpacity
                  onPress={() => { closeSheet(); setTimeout(() => sheetLawyer && openNavigation(sheetLawyer), 300); }}
                  style={{ flex: 1, flexDirection: "row", alignItems: "center", justifyContent: "center", gap: 7, backgroundColor: "#4285F4", borderRadius: 16, paddingVertical: 14 }}
                >
                  <Ionicons name="navigate" size={18} color="#fff" />
                  <Text style={{ fontSize: 13, fontFamily: "inter-semibold", color: "#fff" }}>Naviguer</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  onPress={() => { closeSheet(); setTimeout(() => sheetLawyer && openContact(sheetLawyer), 300); }}
                  style={{ flex: 1, flexDirection: "row", alignItems: "center", justifyContent: "center", gap: 7, backgroundColor: theme.primary, borderRadius: 16, paddingVertical: 14 }}
                >
                  <Ionicons name="chatbubble-ellipses" size={18} color="#fff" />
                  <Text style={{ fontSize: 13, fontFamily: "inter-semibold", color: "#fff" }}>Contacter</Text>
                </TouchableOpacity>
              </View>
            </Pressable>
          </Animated.View>
        </Pressable>
      </Modal>
    </View>
  );
}

// ─── Hub Screen ───────────────────────────────────────────────────────────────

export default function Documents() {
  const theme = useTheme();
  const { t, isRTL } = useTranslation();
  const params = useLocalSearchParams();
  const resumeDocId = params.resumeDocId as string;
  const [activeScreen, setActiveScreen] = useState<Screen>("hub");

  useEffect(() => {
    if (resumeDocId) {
      setActiveScreen("notes");
    }
  }, [resumeDocId]);
  const { lawyers } = useLawyerContext();
  const [indexedArticles, setIndexedArticles] = useState<number | string>("500+");

  useEffect(() => {
    let active = true;
    apiFetch("/health")
      .then(res => res.json())
      .then(data => {
        if (active && data.indexed_articles) {
          setIndexedArticles(data.indexed_articles);
        }
      })
      .catch(err => console.log("Failed to fetch health stats", err));
    return () => { active = false; };
  }, []);

  if (activeScreen === "notes")   return <NotesScreen   onBack={() => setActiveScreen("hub")} resumeDocId={resumeDocId} />;
  if (activeScreen === "lawyers") return <LawyersScreen onBack={() => setActiveScreen("hub")} />;

  const openGazette = async () => {
    // Log activity
    await activityService.addActivity({
      type: "gazette",
      title: t("officialGazette"),
      category: t("officialTexts"),
      url: "https://www.joradp.dz/HAR/Index.htm",
    });

    const url = "https://www.joradp.dz/HAR/Index.htm";
    const canOpen = await Linking.canOpenURL(url);
    if (canOpen) Linking.openURL(url);
    else Alert.alert("Erreur", "Impossible d'ouvrir le lien.");
  };

  const cardAccentLight = theme.primaryLight;
  const cardAccent = theme.primaryMedium;

  return (
    <View style={{ flex: 1, backgroundColor: theme.background }}>
      <ScrollView contentContainerStyle={{ flexGrow: 1 }} showsVerticalScrollIndicator={false}>

        {/* Header */}
        <View style={{ paddingTop: Platform.OS === "ios" ? 56 : 44, paddingHorizontal: 24, paddingBottom: 28, backgroundColor: theme.headerBg, borderBottomLeftRadius: 32, borderBottomRightRadius: 32 }}>
          <Text style={{ fontSize: 11, fontFamily: "inter-semibold", color: theme.primary, letterSpacing: 1.2, textTransform: "uppercase", marginBottom: 6, textAlign: isRTL ? "right" : "left" }}>
            {t("platformLabel")}
          </Text>
          <Text style={{ fontSize: 26, fontFamily: "inter-semibold", color: theme.text, marginBottom: 6, textAlign: isRTL ? "right" : "left" }}>
            {t("legalServices")}
          </Text>
          <Text style={{ fontSize: 13, color: theme.textSecondary, lineHeight: 20, textAlign: isRTL ? "right" : "left" }}>
            {t("legalServicesSub")}
          </Text>
        </View>

        <View style={{ paddingHorizontal: 16, paddingTop: 20, gap: 12 }}>

          {/* Card 1: Journal Officiel */}
          <TouchableOpacity onPress={openGazette} activeOpacity={0.85} style={{ backgroundColor: theme.primary, borderRadius: 22, padding: 20, flexDirection: isRTL ? "row-reverse" : "row", alignItems: "center", gap: 16, shadowColor: theme.primary, shadowOffset: { width: 0, height: 8 }, shadowOpacity: 0.28, shadowRadius: 14, elevation: 8 }}>
            <View style={{ width: 60, height: 60, borderRadius: 18, backgroundColor: "rgba(255,255,255,0.18)", alignItems: "center", justifyContent: "center" }}>
              <Ionicons name="newspaper-outline" size={30} color="#fff" />
            </View>
            <View style={{ flex: 1 }}>
              <View style={{ flexDirection: isRTL ? "row-reverse" : "row", alignItems: "center", gap: 8, marginBottom: 4 }}>
                <Text style={{ fontSize: 17, fontFamily: "inter-semibold", color: "#fff", textAlign: isRTL ? "right" : "left" }}>{t("officialGazette")}</Text>
                <View style={{ backgroundColor: "rgba(255,255,255,0.22)", borderRadius: 8, paddingHorizontal: 7, paddingVertical: 2 }}>
                  <Text style={{ fontSize: 9, fontFamily: "inter-semibold", color: "#fff", letterSpacing: 0.6 }}>{t("official")}</Text>
                </View>
              </View>
              <Text style={{ fontSize: 12, color: "rgba(255,255,255,0.78)", textAlign: isRTL ? "right" : "left" }}>{t("officialGazetteSub")}</Text>
            </View>
            <Ionicons name="open-outline" size={22} color="rgba(255,255,255,0.85)" />
          </TouchableOpacity>

          {/* 2-col: Notes + Lawyers */}
          <View style={{ flexDirection: isRTL ? "row-reverse" : "row", gap: 12 }}>
            {([
              { id: "notes",   icon: "library-outline", titleKey: "legalNotesSvc",  subKey: "legalNotesSvcSub" },
              { id: "lawyers", icon: "people-outline",  titleKey: "findLawyer",     subKey: "findLawyerSub"    },
            ] as const).map(svc => (
              <TouchableOpacity key={svc.id} onPress={() => setActiveScreen(svc.id as Screen)} activeOpacity={0.85}
                style={{ flex: 1, backgroundColor: theme.card, borderRadius: 22, padding: 18, alignItems: "center", gap: 10, shadowColor: theme.shadow, shadowOffset: { width: 0, height: 3 }, shadowOpacity: 0.07, elevation: 3, borderWidth: 1, borderColor: theme.border }}>
                <View style={{ width: 54, height: 54, borderRadius: 17, backgroundColor: cardAccentLight, alignItems: "center", justifyContent: "center", borderWidth: 1.5, borderColor: cardAccent }}>
                  <Ionicons name={svc.icon as any} size={26} color={theme.primary} />
                </View>
                <View style={{ alignItems: "center", gap: 3 }}>
                  <Text style={{ fontSize: 13, fontFamily: "inter-semibold", color: theme.text, textAlign: "center" }}>{t(svc.titleKey)}</Text>
                  <Text style={{ fontSize: 11, color: theme.textSecondary, textAlign: "center", lineHeight: 16 }}>{t(svc.subKey)}</Text>
                </View>
                <View style={{ flexDirection: "row", alignItems: "center", gap: 3 }}>
                  <Text style={{ fontSize: 11, fontFamily: "inter-semibold", color: theme.primary }}>{t("access")}</Text>
                  <Ionicons name={isRTL ? "chevron-back" : "chevron-forward"} size={12} color={theme.primary} />
                </View>
              </TouchableOpacity>
            ))}
          </View>

        </View>

        {/* Stats Banner */}
        <View style={{ margin: 16, marginTop: 22, backgroundColor: theme.card, borderRadius: 22, padding: 20, borderWidth: 1, borderColor: theme.border, shadowColor: theme.shadow, shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.05, elevation: 2 }}>
          <Text style={{ fontSize: 11, fontFamily: "inter-semibold", color: theme.textSecondary, marginBottom: 18, textAlign: "center", textTransform: "uppercase", letterSpacing: 1 }}>
            {t("platformStats")}
          </Text>
          <View style={{ flexDirection: isRTL ? "row-reverse" : "row" }}>
            {[
              { value: `${indexedArticles}`, labelKey: "statsLawTexts",   icon: "document-text-outline" },
              { value: `${LAW_CATEGORIES.length}`,    labelKey: "statsNotesDocs",  icon: "library-outline"       },
              { value: lawyers.length > 0 ? `${lawyers.length}` : "120+", labelKey: "statsLawyersRef", icon: "people-outline"        },
            ].map((stat, i) => (
              <View key={i} style={{ flex: 1, alignItems: "center", gap: 6 }}>
                {i > 0 && <View style={{ position: "absolute", left: 0, top: 8, bottom: 8, width: 1, backgroundColor: theme.divider }} />}
                <View style={{ width: 38, height: 38, borderRadius: 11, backgroundColor: cardAccentLight, alignItems: "center", justifyContent: "center" }}>
                  <Ionicons name={stat.icon as any} size={18} color={theme.primary} />
                </View>
                <Text style={{ fontSize: 20, fontFamily: "inter-semibold", color: theme.text }}>{stat.value}</Text>
                <Text style={{ fontSize: 10, color: theme.textSecondary, textAlign: "center", lineHeight: 14 }}>{t(stat.labelKey as any)}</Text>
              </View>
            ))}
          </View>
        </View>

        {/* Quick Tips */}
        <View style={{ marginHorizontal: 16, marginBottom: 32 }}>
          <Text style={{ fontSize: 11, fontFamily: "inter-semibold", color: theme.textSecondary, marginBottom: 12, textTransform: "uppercase", letterSpacing: 1, textAlign: isRTL ? "right" : "left" }}>
            {t("quickTips")}
          </Text>
          {(["tip1", "tip2", "tip3"] as const).map((tipKey, i) => (
            <View key={i} style={{ flexDirection: isRTL ? "row-reverse" : "row", gap: 12, alignItems: "flex-start", backgroundColor: theme.card, borderRadius: 14, padding: 14, marginBottom: 8, borderWidth: 1, borderColor: theme.border }}>
              <View style={{ width: 32, height: 32, borderRadius: 10, backgroundColor: cardAccentLight, alignItems: "center", justifyContent: "center", marginTop: 1 }}>
                <Ionicons name={["shield-checkmark-outline", "document-attach-outline", "time-outline"][i] as any} size={16} color={theme.primary} />
              </View>
              <Text style={{ flex: 1, fontSize: 13, color: theme.textSecondary, lineHeight: 20, textAlign: isRTL ? "right" : "left" }}>{t(tipKey)}</Text>
            </View>
          ))}
        </View>
      </ScrollView>
    </View>
  );
}
