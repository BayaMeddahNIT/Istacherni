import { useState, useRef, useCallback } from "react";
import {
  Text, View, TextInput, TouchableOpacity,
  ScrollView, Image, FlatList, Dimensions,
  ActivityIndicator, Linking, LayoutAnimation, Keyboard, Platform
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import images from "@/constants/images";
import { router, useFocusEffect } from "expo-router";
import { useTheme, useTranslation } from "@/context/UserContext";
import { apiFetch } from "@/services/api";
import { activityService, RecentActivity } from "@/services/activityService";
import * as WebBrowser from "expo-web-browser";

const { width } = Dimensions.get("window");

export default function Home() {
  const theme = useTheme();
  const { t, isRTL } = useTranslation();
  const [searchQuery, setSearchQuery] = useState("");
  const [activeCard, setActiveCard] = useState(0);
  const flatListRef = useRef<FlatList>(null);

  // Phase 1: Web search state
  const [searchResults, setSearchResults] = useState<any[]>([]);
  const [searchSummary, setSearchSummary] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);
  
  // Recent Activities state
  const [activities, setActivities] = useState<RecentActivity[]>([]);

  // Refresh activities when screen focused
  useFocusEffect(
    useCallback(() => {
      activityService.getActivities().then(setActivities);
    }, [])
  );
  
  // Phase 2: Focus Mode state
  const [isInputFocused, setIsInputFocused] = useState(false);

  const viewableItemsChanged = useRef(({ viewableItems }: any) => {
    if (viewableItems[0]) setActiveCard(viewableItems[0].index);
  }).current;
  const viewConfig = useRef({ viewAreaCoveragePercentThreshold: 50 }).current;

  const handleFocus = () => {
    LayoutAnimation.configureNext(LayoutAnimation.Presets.easeInEaseOut);
    setIsInputFocused(true);
  };

  const handleBlur = () => {
    LayoutAnimation.configureNext(LayoutAnimation.Presets.easeInEaseOut);
    setIsInputFocused(false);
  };

  const handleSearch = async (overrideQuery?: any) => {
    // 🕵️♂️ [object Object] Bug Fix: Extract string if passed an event object
    let queryText = "";
    if (typeof overrideQuery === 'string') {
      queryText = overrideQuery;
    } else if (overrideQuery?.nativeEvent?.text) {
      queryText = overrideQuery.nativeEvent.text;
    } else {
      queryText = searchQuery;
    }

    const safeQuery = String(queryText || "").trim();
    if (!safeQuery || safeQuery === "[object Object]") return;

    // Log activity
    await activityService.addActivity({
      type: "search",
      title: safeQuery,
      query: safeQuery,
    });

    setIsLoading(true);
    setHasSearched(true);
    try {
      const searchUrl = `/api/web-search?question=${encodeURIComponent(safeQuery)}`;
      console.log(`[Frontend Debug] Fetching: ${searchUrl}`);
      const res = await apiFetch(searchUrl);
      if (res.ok) {
        const data = await res.json();
        setSearchResults(data.results || []);
        setSearchSummary(data.summary || "");
      } else {
        console.error("API error", res.status);
      }
    } catch (err: any) {
      if (err?.name === "AbortError") return;
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleResetHome = () => {
    LayoutAnimation.configureNext(LayoutAnimation.Presets.easeInEaseOut);
    setHasSearched(false);
    setSearchResults([]);
    setSearchSummary("");
    setSearchQuery("");
  };

  const openLink = async (url: string) => {
    if (url) {
      await WebBrowser.openBrowserAsync(url, {
        presentationStyle: WebBrowser.WebBrowserPresentationStyle.MODAL,
        toolbarColor: theme.background,
        controlsColor: theme.primary,
      });
    }
  };

  return (
    <View style={{ flex: 1, backgroundColor: theme.background }}>
      <ScrollView contentContainerStyle={{ flexGrow: 1 }} showsVerticalScrollIndicator={false}>

        {/* Top Banner */}
        {!(isInputFocused || hasSearched) && (
          <View style={{
            height: 280, backgroundColor: theme.headerBg,
            borderBottomLeftRadius: 30, borderBottomRightRadius: 30,
            alignItems: "center", justifyContent: "center",
          }}>
            <Image source={images.logo} style={{ width: 180, height: 120 }} resizeMode="contain" />
          </View>
        )}

        {/* Search */}
        <View style={{ paddingHorizontal: 24, marginTop: (isInputFocused || hasSearched) ? Platform.OS === 'ios' ? 60 : 40 : -28 }}>
          <View style={{
            backgroundColor: theme.card, borderRadius: 30,
            paddingHorizontal: 20, height: 56,
            flexDirection: isRTL ? "row-reverse" : "row",
            alignItems: "center",
            shadowColor: "#000", shadowOffset: { width: 0, height: 4 },
            shadowOpacity: 0.1, shadowRadius: 12, elevation: 5,
            borderWidth: 1, borderColor: theme.border,
          }}>
            {hasSearched ? (
              <TouchableOpacity onPress={handleResetHome} style={{ marginRight: isRTL ? 0 : 10, marginLeft: isRTL ? 10 : 0, padding: 2 }}>
                <Ionicons name="arrow-back" size={24} color={theme.primary} />
              </TouchableOpacity>
            ) : (
              <Ionicons name="search" size={22} color={theme.textMuted} style={{ marginRight: isRTL ? 0 : 12, marginLeft: isRTL ? 12 : 0 }} />
            )}
            <TextInput
              placeholder="Ask your legal question..."
              placeholderTextColor={theme.textMuted}
              value={searchQuery}
              onChangeText={setSearchQuery}
              onFocus={handleFocus}
              onBlur={handleBlur}
              style={{ flex: 1, fontSize: 16, color: theme.text, textAlign: isRTL ? "right" : "left", height: "100%" }}
              returnKeyType="search"
              onSubmitEditing={() => handleSearch(searchQuery)}
            />
            {searchQuery.length > 0 ? (
              <TouchableOpacity onPress={() => setSearchQuery("")} style={{ padding: 4, marginLeft: isRTL ? 0 : 10, marginRight: isRTL ? 10 : 0 }}>
                <Ionicons name="close-circle" size={20} color={theme.textMuted} />
              </TouchableOpacity>
            ) : (
              <TouchableOpacity style={{ padding: 4, marginLeft: isRTL ? 0 : 10, marginRight: isRTL ? 10 : 0 }}>
                <Ionicons name="mic" size={22} color={theme.primary} />
              </TouchableOpacity>
            )}
          </View>
        </View>


        {/* Search Results or Recent Researches */}
        {hasSearched ? (
          <View style={{ marginTop: 24, paddingHorizontal: 24 }}>
            {isLoading ? (
              <ActivityIndicator size="large" color={theme.primary} style={{ marginTop: 40 }} />
            ) : searchResults.length > 0 ? (
              <View style={{ gap: 16, paddingBottom: 40 }}>
                {searchSummary ? (
                  <View style={{ backgroundColor: theme.primary + "15", padding: 16, borderRadius: 16, borderWidth: 1, borderColor: theme.primary + "40", marginBottom: 8 }}>
                    <View style={{ flexDirection: "row", alignItems: "center", marginBottom: 8 }}>
                      <Ionicons name="sparkles" size={16} color={theme.primary} style={{ marginRight: isRTL ? 0 : 8, marginLeft: isRTL ? 8 : 0 }} />
                      <Text style={{ fontSize: 13, fontFamily: "inter-semibold", color: theme.primary }}>Aperçu de l'IA (Qwen2)</Text>
                    </View>
                    <Text style={{ fontSize: 14, color: theme.text, lineHeight: 22, textAlign: isRTL ? "right" : "left" }}>
                      {searchSummary}
                    </Text>
                  </View>
                ) : null}
                
                <Text style={{ fontSize: 16, fontFamily: "inter-semibold", color: theme.text, marginBottom: 8, textAlign: isRTL ? "right" : "left" }}>
                  Résultats de la recherche
                </Text>
                {searchResults.map((item, index) => (
                  <TouchableOpacity 
                    key={index} 
                    onPress={() => openLink(item.link)}
                    style={{
                      backgroundColor: theme.card, borderRadius: 16, padding: 16,
                      shadowColor: theme.shadow, shadowOffset: { width: 0, height: 1 },
                      shadowOpacity: 0.05, shadowRadius: 4, elevation: 2,
                    }}
                  >
                    <Text style={{ fontSize: 11, color: theme.primary, fontFamily: "inter-semibold", marginBottom: 4 }} numberOfLines={1}>
                      {item.source}
                    </Text>
                    <Text style={{ fontSize: 15, fontFamily: "inter-semibold", color: theme.text, marginBottom: 6 }}>
                      {item.title}
                    </Text>
                    <Text style={{ fontSize: 13, color: theme.textSecondary, lineHeight: 18 }} numberOfLines={3}>
                      {item.snippet}
                    </Text>
                  </TouchableOpacity>
                ))}
              </View>
            ) : (
              <Text style={{ textAlign: "center", color: theme.textSecondary, marginTop: 40 }}>
                Aucun résultat trouvé.
              </Text>
            )}
          </View>
        ) : (!isInputFocused && activities.length > 0) ? (
          <>
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
              data={activities}
              horizontal
              pagingEnabled
              showsHorizontalScrollIndicator={false}
              bounces={false}
              keyExtractor={(item) => item.id}
              onViewableItemsChanged={viewableItemsChanged}
              viewabilityConfig={viewConfig}
              renderItem={({ item }) => (
                <View style={{ width: width - 48, marginHorizontal: 24 }}>
                  <TouchableOpacity 
                    activeOpacity={0.9} 
                    onPress={() => {
                      if (item.type === "search" && item.query) {
                        setSearchQuery(item.query);
                        handleSearch(item.query);
                      } else if (item.type === "document" && item.documentId) {
                        router.push({ pathname: "/documents", params: { resumeDocId: item.documentId } });
                      } else if (item.type === "gazette" && item.url) {
                        Linking.openURL(item.url);
                      }
                    }}
                    style={{
                      backgroundColor: theme.card, borderRadius: 18, padding: 16,
                      flexDirection: isRTL ? "row-reverse" : "row",
                      shadowColor: theme.shadow, shadowOffset: { width: 0, height: 2 },
                      shadowOpacity: 0.08, shadowRadius: 8, elevation: 3,
                    }}
                  >
                    <View style={{
                      width: 100, height: 120, borderRadius: 12,
                      backgroundColor: theme.pillBg, overflow: "hidden", 
                      marginRight: isRTL ? 0 : 14, marginLeft: isRTL ? 14 : 0,
                      alignItems: "center", justifyContent: "center",
                    }}>
                      <Ionicons 
                        name={item.type === "search" ? "search" : item.type === "gazette" ? "newspaper" : "document-text"} 
                        size={40} 
                        color={theme.primary} 
                      />
                    </View>
                    <View style={{ flex: 1 }}>
                      <Text style={{ 
                        fontSize: 12, 
                        fontFamily: "inter-semibold", 
                        color: theme.primary, 
                        marginBottom: 4,
                        textAlign: isRTL ? "right" : "left" 
                      }}>
                        {item.type === "search" ? t("recentSearch") : item.type === "gazette" ? t("officialGazette") : t("openedDocument")}
                      </Text>
                      <Text style={{ 
                        fontSize: 15, 
                        fontFamily: "inter-semibold", 
                        color: theme.text, 
                        marginBottom: 4,
                        textAlign: isRTL ? "right" : "left"
                      }} numberOfLines={2}>
                        {item.title}
                      </Text>
                      {item.category && (
                        <Text style={{ 
                          fontSize: 13, 
                          color: theme.textSecondary, 
                          lineHeight: 18,
                          textAlign: isRTL ? "right" : "left"
                        }} numberOfLines={2}>
                          {item.category}
                        </Text>
                      )}
                      <Text style={{ 
                        fontSize: 11, 
                        color: theme.textMuted, 
                        marginTop: 8,
                        textAlign: isRTL ? "right" : "left"
                      }}>
                        {new Date(item.timestamp).toLocaleDateString()}
                      </Text>
                    </View>
                  </TouchableOpacity>
                </View>
              )}
            />

            <View style={{ flexDirection: "row", justifyContent: "center", marginTop: 14, marginBottom: 20 }}>
              {activities.map((_, index) => (
                <View key={index} style={{
                  width: 8, height: 8, borderRadius: 4, marginHorizontal: 4,
                  backgroundColor: activeCard === index ? theme.primary : theme.divider,
                }} />
              ))}
            </View>
          </>
        ) : null}
      </ScrollView>
    </View>
  );
}
