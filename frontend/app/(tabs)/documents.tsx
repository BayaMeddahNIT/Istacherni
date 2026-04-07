import { useState, useEffect } from "react";
import {
  Text, View, ScrollView, TouchableOpacity, FlatList,
  TextInput, Linking, Alert, Platform,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import * as Location from "expo-location";
import { router } from "expo-router";
import { useTheme, useTranslation } from "@/context/UserContext";

// ─── Types ────────────────────────────────────────────────────────────────────

type Screen = "hub" | "notes" | "lawyers";

interface Lawyer {
  id: string;
  name: string;
  specialty: string;
  phone: string;
  address: string;
  city: string;
  rating: number;
  experience: number;
  avatar: string;
  available: boolean;
}

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

// ─── Lawyer Data ──────────────────────────────────────────────────────────────

const LAWYERS: Lawyer[] = [
  { id: "1", name: "Maître Karim Boudiaf",  specialty: "Droit du Travail",    phone: "+213 555 123 456", address: "12 Rue Didouche Mourad",  city: "Alger",       rating: 4.8, experience: 15, avatar: "KB", available: true  },
  { id: "2", name: "Maître Samira Hadj",    specialty: "Droit de la Famille", phone: "+213 555 234 567", address: "34 Avenue Krim Belkacem", city: "Alger",       rating: 4.9, experience: 12, avatar: "SH", available: true  },
  { id: "3", name: "Maître Youcef Ziani",   specialty: "Droit Commercial",    phone: "+213 555 345 678", address: "7 Rue Ben M'hidi",         city: "Oran",        rating: 4.7, experience: 20, avatar: "YZ", available: false },
  { id: "4", name: "Maître Nadia Bensalem", specialty: "Droit Pénal",         phone: "+213 555 456 789", address: "22 Rue de la Liberté",     city: "Constantine", rating: 4.6, experience: 8,  avatar: "NB", available: true  },
  { id: "5", name: "Maître Riad Meziane",   specialty: "Droit Immobilier",    phone: "+213 555 567 890", address: "5 Place du 1er Novembre",  city: "Alger",       rating: 4.5, experience: 10, avatar: "RM", available: true  },
  { id: "6", name: "Maître Fatima Cherif",  specialty: "Droit Administratif", phone: "+213 555 678 901", address: "18 Bd Colonel Amirouche",  city: "Tizi Ouzou",  rating: 4.7, experience: 14, avatar: "FC", available: false },
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

function NotesScreen({ onBack }: { onBack: () => void }) {
  const theme = useTheme();
  const { t, isRTL } = useTranslation();
  const [loading, setLoading] = useState<string | null>(null);

  const openPdf = async (cat: typeof LAW_CATEGORIES[0]) => {
    setLoading(cat.id);
    try {
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

function LawyersScreen({ onBack }: { onBack: () => void }) {
  const theme = useTheme();
  const { t, isRTL } = useTranslation();
  const [query, setQuery] = useState("");
  const [locationGranted, setLocationGranted] = useState(false);

  useEffect(() => {
    Location.requestForegroundPermissionsAsync().then(({ granted }) => setLocationGranted(granted));
  }, []);

  const filtered = LAWYERS.filter(l =>
    l.name.toLowerCase().includes(query.toLowerCase()) ||
    l.specialty.toLowerCase().includes(query.toLowerCase()) ||
    l.city.toLowerCase().includes(query.toLowerCase())
  );

  return (
    <View style={{ flex: 1, backgroundColor: theme.background }}>
      {/* Header */}
      <View style={{
        paddingTop: Platform.OS === "ios" ? 56 : 44,
        paddingHorizontal: 16, paddingBottom: 16,
        backgroundColor: theme.headerBg,
        borderBottomLeftRadius: 24, borderBottomRightRadius: 24,
      }}>
        <View style={{ flexDirection: isRTL ? "row-reverse" : "row", alignItems: "center", gap: 12, marginBottom: 16 }}>
          <TouchableOpacity onPress={onBack} style={{ width: 38, height: 38, borderRadius: 12, backgroundColor: theme.primaryMedium, alignItems: "center", justifyContent: "center" }}>
            <Ionicons name={isRTL ? "arrow-forward" : "arrow-back"} size={20} color={theme.primary} />
          </TouchableOpacity>
          <View style={{ flex: 1 }}>
            <Text style={{ fontSize: 18, fontFamily: "inter-semibold", color: theme.text, textAlign: isRTL ? "right" : "left" }}>{t("lawyersAvailable")}</Text>
            <Text style={{ fontSize: 12, color: theme.textSecondary, marginTop: 1, textAlign: isRTL ? "right" : "left" }}>
              {locationGranted ? `📍 ${t("nearYou")}` : `${filtered.length} ${t("lawyersReferenced")}`}
            </Text>
          </View>
        </View>
        <View style={{ flexDirection: isRTL ? "row-reverse" : "row", alignItems: "center", backgroundColor: theme.card, borderRadius: 14, paddingHorizontal: 14, height: 44, gap: 8 }}>
          <Ionicons name="search" size={18} color={theme.textMuted} />
          <TextInput
            placeholder={t("searchLawyerPlaceholder")}
            placeholderTextColor={theme.textMuted}
            value={query}
            onChangeText={setQuery}
            style={{ flex: 1, fontSize: 14, color: theme.text, textAlign: isRTL ? "right" : "left" }}
          />
          {query.length > 0 && (
            <TouchableOpacity onPress={() => setQuery("")}>
              <Ionicons name="close-circle" size={18} color={theme.textMuted} />
            </TouchableOpacity>
          )}
        </View>
      </View>

      <FlatList
        data={filtered}
        keyExtractor={l => l.id}
        contentContainerStyle={{ padding: 16, gap: 12 }}
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
              {/* Avatar */}
              <View style={{ width: 56, height: 56, borderRadius: 18, backgroundColor: theme.primaryLight, borderWidth: 2, borderColor: theme.primaryMedium, alignItems: "center", justifyContent: "center" }}>
                <Text style={{ fontSize: 16, fontFamily: "inter-semibold", color: theme.primary }}>{item.avatar}</Text>
              </View>

              <View style={{ flex: 1 }}>
                <View style={{ flexDirection: isRTL ? "row-reverse" : "row", alignItems: "center", justifyContent: "space-between" }}>
                  <Text style={{ fontSize: 15, fontFamily: "inter-semibold", color: theme.text, flex: 1, textAlign: isRTL ? "right" : "left" }}>{item.name}</Text>
                  <View style={{ backgroundColor: item.available ? theme.success + "18" : theme.pillBg, borderRadius: 8, paddingHorizontal: 8, paddingVertical: 3 }}>
                    <Text style={{ fontSize: 10, fontFamily: "inter-semibold", color: item.available ? theme.success : theme.textMuted }}>
                      {item.available ? t("available") : t("busy")}
                    </Text>
                  </View>
                </View>

                <View style={{ backgroundColor: theme.primaryLight, borderRadius: 8, paddingHorizontal: 8, paddingVertical: 3, alignSelf: isRTL ? "flex-end" : "flex-start", marginTop: 4 }}>
                  <Text style={{ fontSize: 11, fontFamily: "inter-semibold", color: theme.primary }}>{item.specialty}</Text>
                </View>

                <View style={{ marginTop: 8, gap: 3 }}>
                  <View style={{ flexDirection: isRTL ? "row-reverse" : "row", alignItems: "center", gap: 6 }}>
                    <Ionicons name="location-outline" size={12} color={theme.textMuted} />
                    <Text style={{ fontSize: 12, color: theme.textSecondary }} numberOfLines={1}>{item.address}, {item.city}</Text>
                  </View>
                  <View style={{ flexDirection: isRTL ? "row-reverse" : "row", alignItems: "center", gap: 6 }}>
                    <Ionicons name="briefcase-outline" size={12} color={theme.textMuted} />
                    <Text style={{ fontSize: 12, color: theme.textSecondary }}>{item.experience} {t("experience")}</Text>
                  </View>
                  <View style={{ flexDirection: isRTL ? "row-reverse" : "row", alignItems: "center", gap: 3 }}>
                    {[1,2,3,4,5].map(s => (
                      <Ionicons key={s} name={s <= Math.floor(item.rating) ? "star" : "star-outline"} size={12} color="#F2C94C" />
                    ))}
                    <Text style={{ fontSize: 11, color: theme.textMuted, marginLeft: 4 }}>{item.rating}</Text>
                  </View>
                </View>
              </View>
            </View>

            {/* Action buttons */}
            <View style={{ flexDirection: isRTL ? "row-reverse" : "row", gap: 8, marginTop: 14 }}>
              <TouchableOpacity onPress={() => Linking.openURL(`tel:${item.phone.replace(/\s/g, "")}`)} style={{ flex: 1, flexDirection: "row", alignItems: "center", justifyContent: "center", gap: 5, backgroundColor: theme.success + "14", borderRadius: 12, paddingVertical: 10, borderWidth: 1, borderColor: theme.success + "30" }}>
                <Ionicons name="call-outline" size={15} color={theme.success} />
                <Text style={{ fontSize: 12, fontFamily: "inter-semibold", color: theme.success }}>{t("call")}</Text>
              </TouchableOpacity>
              <TouchableOpacity onPress={() => Linking.openURL(`https://wa.me/${item.phone.replace(/[^0-9]/g, "")}`)} style={{ flex: 1, flexDirection: "row", alignItems: "center", justifyContent: "center", gap: 5, backgroundColor: "#25D36614", borderRadius: 12, paddingVertical: 10, borderWidth: 1, borderColor: "#25D36630" }}>
                <Ionicons name="logo-whatsapp" size={15} color="#25D366" />
                <Text style={{ fontSize: 12, fontFamily: "inter-semibold", color: "#25D366" }}>WhatsApp</Text>
              </TouchableOpacity>
              <TouchableOpacity onPress={() => Alert.alert(t("appointment"), item.name)} style={{ flex: 1, flexDirection: "row", alignItems: "center", justifyContent: "center", gap: 5, backgroundColor: theme.primaryLight, borderRadius: 12, paddingVertical: 10, borderWidth: 1, borderColor: theme.primaryMedium }}>
                <Ionicons name="calendar-outline" size={15} color={theme.primary} />
                <Text style={{ fontSize: 12, fontFamily: "inter-semibold", color: theme.primary }}>{t("appointment")}</Text>
              </TouchableOpacity>
            </View>
          </View>
        )}
      />
    </View>
  );
}

// ─── Hub Screen ───────────────────────────────────────────────────────────────

export default function Documents() {
  const theme = useTheme();
  const { t, isRTL } = useTranslation();
  const [activeScreen, setActiveScreen] = useState<Screen>("hub");

  if (activeScreen === "notes")   return <NotesScreen   onBack={() => setActiveScreen("hub")} />;
  if (activeScreen === "lawyers") return <LawyersScreen onBack={() => setActiveScreen("hub")} />;

  const openGazette = async () => {
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

          {/* Card 3: Contracts AI */}
          <TouchableOpacity onPress={() => router.push("/contract-ai")} activeOpacity={0.85}
            style={{ backgroundColor: theme.card, borderRadius: 22, padding: 20, flexDirection: isRTL ? "row-reverse" : "row", alignItems: "center", gap: 16, shadowColor: theme.shadow, shadowOffset: { width: 0, height: 4 }, shadowOpacity: 0.08, elevation: 4, borderWidth: 1.5, borderColor: cardAccent }}>
            <View style={{ width: 60, height: 60, borderRadius: 18, backgroundColor: cardAccentLight, alignItems: "center", justifyContent: "center", borderWidth: 1.5, borderColor: cardAccent }}>
              <Ionicons name="sparkles" size={28} color={theme.primary} />
            </View>
            <View style={{ flex: 1 }}>
              <View style={{ flexDirection: isRTL ? "row-reverse" : "row", alignItems: "center", gap: 8, marginBottom: 4 }}>
                <Text style={{ fontSize: 16, fontFamily: "inter-semibold", color: theme.text, textAlign: isRTL ? "right" : "left" }}>{t("contractsAI")}</Text>
                <View style={{ backgroundColor: cardAccentLight, borderRadius: 8, paddingHorizontal: 7, paddingVertical: 2 }}>
                  <Text style={{ fontSize: 9, fontFamily: "inter-semibold", color: theme.primary, letterSpacing: 0.5 }}>IA</Text>
                </View>
              </View>
              <Text style={{ fontSize: 12, color: theme.textSecondary, textAlign: isRTL ? "right" : "left" }}>{t("contractsAISub")}</Text>
            </View>
            <Ionicons name={isRTL ? "chevron-back" : "chevron-forward"} size={20} color={theme.primary} />
          </TouchableOpacity>
        </View>

        {/* Stats Banner */}
        <View style={{ margin: 16, marginTop: 22, backgroundColor: theme.card, borderRadius: 22, padding: 20, borderWidth: 1, borderColor: theme.border, shadowColor: theme.shadow, shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.05, elevation: 2 }}>
          <Text style={{ fontSize: 11, fontFamily: "inter-semibold", color: theme.textSecondary, marginBottom: 18, textAlign: "center", textTransform: "uppercase", letterSpacing: 1 }}>
            {t("platformStats")}
          </Text>
          <View style={{ flexDirection: isRTL ? "row-reverse" : "row" }}>
            {[
              { value: "500+", labelKey: "statsLawTexts",   icon: "document-text-outline" },
              { value: "4",    labelKey: "statsNotesDocs",  icon: "library-outline"       },
              { value: "120+", labelKey: "statsLawyersRef", icon: "people-outline"        },
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
