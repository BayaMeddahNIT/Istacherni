import React, { useState, useRef, useEffect } from "react";
import {
  View,
  Text,
  StyleSheet,
  Dimensions,
  FlatList,
  Image,
  TouchableOpacity,
  SafeAreaView,
  StatusBar,
  Animated,
} from "react-native";
import { router } from "expo-router";
import { useUser } from "@/context/UserContext";
import AuthControls from "@/components/AuthControls";
import images from "@/constants/images";

const { width, height } = Dimensions.get("window");

// ── Slide Data ──────────────────────────────────────────────────────────────

interface OnboardingSlide {
  id: string;
  title?: string;
  description?: string;
  image?: any;
  isSplash?: boolean;
}

const slides: OnboardingSlide[] = [
  {
    id: "0",
    isSplash: true,
  },
  {
    id: "1",
    title: "onboardingTitle1",
    description: "onboardingDesc1",
    image: require("@/assets/images/1.png"),
  },
  {
    id: "2",
    title: "onboardingTitle2",
    description: "onboardingDesc2",
    image: require("@/assets/images/2.png"),
  },
  {
    id: "3",
    title: "onboardingTitle3",
    description: "onboardingDesc3",
    image: require("@/assets/images/3.png"),
  },
];

// ── Splash Slide Component (with animated logo) ──────────────────────────────

function SplashSlide({
  theme,
  t,
  onNext,
}: {
  theme: any;
  t: any;
  onNext: () => void;
}) {
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const scaleAnim = useRef(new Animated.Value(0.8)).current;

  useEffect(() => {
    Animated.parallel([
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 900,
        useNativeDriver: true,
      }),
      Animated.spring(scaleAnim, {
        toValue: 1,
        tension: 60,
        friction: 8,
        useNativeDriver: true,
      }),
    ]).start();
  }, []);

  return (
    <View style={[styles.slide, { width, backgroundColor: theme.background }]}>
      {/* Animated Logo */}
      <Animated.View
        style={[
          styles.logoWrapper,
          { opacity: fadeAnim, transform: [{ scale: scaleAnim }] },
        ]}
      >
        <Image
          source={images.appLogo}
          style={styles.splashLogo}
          resizeMode="contain"
        />
      </Animated.View>

      {/* Start button */}
      <View style={styles.splashContent}>
        <TouchableOpacity
          onPress={onNext}
          activeOpacity={0.8}
          style={[styles.primaryButton, { backgroundColor: theme.primary }]}
        >
          <Text style={styles.primaryButtonText}>{t("onboardingStart")}</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

// ── Main Component ──────────────────────────────────────────────────────────

export default function OnboardingScreen() {
  const { theme, t, isRTL, language, setLanguage, darkMode, setDarkMode, setHasCompletedOnboarding } = useUser();
  const [currentIndex, setCurrentIndex] = useState(0);
  const scrollX = useRef(new Animated.Value(0)).current;
  const slidesRef = useRef<FlatList>(null);

  const viewableItemsChanged = useRef(({ viewableItems }: any) => {
    setCurrentIndex(viewableItems[0]?.index || 0);
  }).current;

  const viewConfig = useRef({ viewAreaCoveragePercentThreshold: 50 }).current;

  const handleNext = () => {
    if (currentIndex < slides.length - 1) {
      slidesRef.current?.scrollToIndex({ index: currentIndex + 1 });
    } else {
      completeOnboarding();
    }
  };

  const completeOnboarding = () => {
    setHasCompletedOnboarding(true);
    router.replace("/(auth)/log-in");
  };

  // ── Render Items ──────────────────────────────────────────────────────────

  const renderSlide = ({ item }: { item: OnboardingSlide }) => {
    if (item.isSplash) {
      return <SplashSlide theme={theme} t={t} onNext={handleNext} />;
    }

    return (
      <View style={[styles.slide, { width, backgroundColor: theme.background }]}>
        <View style={styles.imageContainer}>
          <Image source={item.image} style={styles.image} resizeMode="contain" />
        </View>

        <View style={styles.textContainer}>
          <Text style={[styles.title, { color: theme.text, textAlign: "center" }]}>
            {t(item.title as any)}
          </Text>
          <Text style={[styles.description, { color: theme.textSecondary, textAlign: "center" }]}>
            {t(item.description as any)}
          </Text>
        </View>

        <View style={styles.footer}>
          <TouchableOpacity
            onPress={handleNext}
            activeOpacity={0.8}
            style={[styles.primaryButton, { backgroundColor: theme.primary }]}
          >
            <Text style={styles.primaryButtonText}>
              {currentIndex === slides.length - 1 ? t("onboardingStart") : t("onboardingNext")}
            </Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  };

  // ── UI ────────────────────────────────────────────────────────────────────

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: theme.background }]}>
      <StatusBar barStyle={darkMode ? "light-content" : "dark-content"} />
      
      <AuthControls />

      <FlatList
        data={slides}
        renderItem={renderSlide}
        horizontal
        showsHorizontalScrollIndicator={false}
        pagingEnabled
        bounces={false}
        keyExtractor={(item) => item.id}
        onScroll={Animated.event([{ nativeEvent: { contentOffset: { x: scrollX } } }], {
          useNativeDriver: false,
        })}
        onViewableItemsChanged={viewableItemsChanged}
        viewabilityConfig={viewConfig}
        ref={slidesRef}
        scrollEventThrottle={32}
      />

      {/* Pagination Dots */}
      <View style={styles.paginationContainer}>
        {slides.map((_, i) => {
          const inputRange = [(i - 1) * width, i * width, (i + 1) * width];
          const dotWidth = scrollX.interpolate({
            inputRange,
            outputRange: [10, 24, 10],
            extrapolate: "clamp",
          });
          const opacity = scrollX.interpolate({
            inputRange,
            outputRange: [0.3, 1, 0.3],
            extrapolate: "clamp",
          });

          return (
            <Animated.View
              key={i.toString()}
              style={[
                styles.dot,
                { width: dotWidth, opacity, backgroundColor: theme.primary },
              ]}
            />
          );
        })}
      </View>

      {/* Skip Button (not on splash) */}
      {currentIndex > 0 && currentIndex < slides.length - 1 && (
        <TouchableOpacity style={styles.skipButton} onPress={completeOnboarding}>
          <Text style={[styles.skipText, { color: theme.textSecondary }]}>{t("onboardingSkip")}</Text>
        </TouchableOpacity>
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  slide: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
  },
  // Splash styles
  logoWrapper: {
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 60,
  },
  splashLogo: {
    width: width * 0.65,
    height: width * 0.65,
  },
  splashContent: {
    width: "100%",
    paddingHorizontal: 40,
    position: "absolute",
    bottom: 80,
  },
  // Feature slide styles
  imageContainer: {
    flex: 0.5,
    justifyContent: "center",
    alignItems: "center",
    width: "80%",
  },
  image: {
    width: "100%",
    height: "100%",
  },
  textContainer: {
    flex: 0.3,
    paddingHorizontal: 40,
    alignItems: "center",
  },
  title: {
    fontSize: 28,
    fontWeight: "bold",
    marginBottom: 15,
  },
  description: {
    fontSize: 16,
    lineHeight: 24,
  },
  footer: {
    flex: 0.2,
    justifyContent: "center",
    width: "100%",
    paddingHorizontal: 40,
  },
  primaryButton: {
    height: 56,
    borderRadius: 28,
    alignItems: "center",
    justifyContent: "center",
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.2,
    shadowRadius: 5,
    elevation: 8,
  },
  primaryButtonText: {
    color: "#fff",
    fontSize: 18,
    fontWeight: "bold",
  },
  paginationContainer: {
    flexDirection: "row",
    height: 64,
    justifyContent: "center",
    alignItems: "center",
    position: "absolute",
    bottom: 20,
    width: "100%",
  },
  dot: {
    height: 10,
    borderRadius: 5,
    marginHorizontal: 4,
  },
  skipButton: {
    position: "absolute",
    bottom: 40,
    right: 40,
  },
  skipText: {
    fontSize: 16,
    fontWeight: "600",
  },
  header: {
    paddingHorizontal: 20,
    paddingVertical: 10,
    justifyContent: "space-between",
    alignItems: "center",
    zIndex: 10,
  },
  iconButton: {
    flexDirection: "row",
    alignItems: "center",
    padding: 8,
    borderRadius: 20,
  },
  iconText: {
    marginLeft: 6,
    fontWeight: "bold",
    fontSize: 14,
  },
});
