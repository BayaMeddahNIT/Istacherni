import { useState, useRef } from "react";
import {
  Text, View, TextInput, TouchableOpacity,
  ScrollView, Image, FlatList, Dimensions,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import images from "@/constants/images";
import { router } from "expo-router";
import { useTheme, useTranslation } from "@/context/UserContext";

const { width } = Dimensions.get("window");

const recentResearches = [
  {
    id: "1",
    title: "Droit du travail algérien",
    description: "Le droit du travail algérien encadre les relations entre employeurs et travailleurs. Il couvre les contrats, les congés, les licenciements et les droits syndicaux.",
  },
  {
    id: "2",
    title: "Code pénal - Article 263",
    description: "Dispositions relatives aux infractions contre les personnes. Peines et sanctions prévues par la législation algérienne en matière de droit pénal.",
  },
  {
    id: "3",
    title: "Code de commerce – Faillite",
    description: "Procédures relatives à l'insolvabilité et à la liquidation des entreprises selon le Code de Commerce algérien (Ordonnance n° 75-59).",
  },
];

export default function Home() {
  const theme = useTheme();
  const { t, isRTL } = useTranslation();
  const [searchQuery, setSearchQuery] = useState("");
  const [activeCard, setActiveCard] = useState(0);
  const flatListRef = useRef<FlatList>(null);

  const viewableItemsChanged = useRef(({ viewableItems }: any) => {
    if (viewableItems[0]) setActiveCard(viewableItems[0].index);
  }).current;
  const viewConfig = useRef({ viewAreaCoveragePercentThreshold: 50 }).current;

  return (
    <View style={{ flex: 1, backgroundColor: theme.background }}>
      <ScrollView contentContainerStyle={{ flexGrow: 1 }} showsVerticalScrollIndicator={false}>

        {/* Top Banner */}
        <View style={{
          height: 280, backgroundColor: theme.headerBg,
          borderBottomLeftRadius: 30, borderBottomRightRadius: 30,
          alignItems: "center", justifyContent: "center",
        }}>
          <Image source={images.logo} style={{ width: 180, height: 120 }} resizeMode="contain" />
        </View>

        {/* Search */}
        <View style={{ paddingHorizontal: 24, marginTop: -28 }}>
          <View style={{
            backgroundColor: theme.card, borderRadius: 30,
            paddingHorizontal: 20, height: 56,
            flexDirection: isRTL ? "row-reverse" : "row",
            alignItems: "center",
            shadowColor: theme.shadow, shadowOffset: { width: 0, height: 2 },
            shadowOpacity: 0.08, shadowRadius: 8, elevation: 3,
          }}>
            <TextInput
              placeholder={t("searchPlaceholder")}
              placeholderTextColor={theme.textMuted}
              value={searchQuery}
              onChangeText={setSearchQuery}
              style={{ flex: 1, fontSize: 15, color: theme.text, textAlign: isRTL ? "right" : "left" }}
              returnKeyType="search"
            />
            <Ionicons name="search" size={22} color={theme.primary} />
          </View>
        </View>


        {/* Recent Researches */}
        <View style={{ marginTop: 24, paddingHorizontal: 24 }}>
          <Text style={{
            fontSize: 14, fontFamily: "inter-medium", color: theme.textSecondary,
            marginBottom: 14, textAlign: isRTL ? "right" : "left",
          }}>
            {t("recentResearches")}
          </Text>
        </View>

        <FlatList
          ref={flatListRef}
          data={recentResearches}
          horizontal
          pagingEnabled
          showsHorizontalScrollIndicator={false}
          bounces={false}
          keyExtractor={(item) => item.id}
          onViewableItemsChanged={viewableItemsChanged}
          viewabilityConfig={viewConfig}
          renderItem={({ item }) => (
            <View style={{ width: width - 48, marginHorizontal: 24 }}>
              <TouchableOpacity activeOpacity={0.9} style={{
                backgroundColor: theme.card, borderRadius: 18, padding: 16,
                flexDirection: "row",
                shadowColor: theme.shadow, shadowOffset: { width: 0, height: 2 },
                shadowOpacity: 0.08, shadowRadius: 8, elevation: 3,
              }}>
                <View style={{
                  width: 100, height: 120, borderRadius: 12,
                  backgroundColor: theme.pillBg, overflow: "hidden", marginRight: 14,
                  alignItems: "center", justifyContent: "center",
                }}>
                  <Ionicons name="document-text" size={40} color={theme.primary} />
                </View>
                <View style={{ flex: 1 }}>
                  <Text style={{ fontSize: 15, fontFamily: "inter-semibold", color: theme.text, marginBottom: 8 }} numberOfLines={2}>
                    {item.title}
                  </Text>
                  <Text style={{ fontSize: 13, color: theme.textSecondary, lineHeight: 18 }} numberOfLines={5}>
                    {item.description}
                  </Text>
                </View>
              </TouchableOpacity>
            </View>
          )}
        />

        {/* Carousel Dots */}
        <View style={{ flexDirection: "row", justifyContent: "center", marginTop: 14, marginBottom: 20 }}>
          {recentResearches.map((_, index) => (
            <View key={index} style={{
              width: 8, height: 8, borderRadius: 4, marginHorizontal: 4,
              backgroundColor: activeCard === index ? theme.primary : theme.divider,
            }} />
          ))}
        </View>
      </ScrollView>
    </View>
  );
}
