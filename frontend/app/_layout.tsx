import { useEffect } from "react";
import { ActivityIndicator, View, Image } from "react-native";
import { router, Stack } from "expo-router";
import "@/global.css";
import { UserProvider, useUser } from "@/context/UserContext";
import { AuthProvider, useAuth } from "@/context/AuthContext";
import { LawyerProvider } from "@/context/LawyerContext";
import images from "@/constants/images";

// ── Inner layout: reads auth state and enforces the guard ─────────────────────
function RootNavigator() {
  const { isAuthenticated, isLoading } = useAuth();
  const { hasCompletedOnboarding } = useUser();

  useEffect(() => {
    if (isLoading) return; // wait until AsyncStorage check is done

    if (!hasCompletedOnboarding) {
      router.replace("/(onboarding)");
      return;
    }

    if (isAuthenticated) {
      // Already logged in — go straight to the app
      router.replace("/(tabs)/home");
    } else {
      // No valid token — force login
      router.replace("/(auth)/log-in");
    }
  }, [isAuthenticated, isLoading, hasCompletedOnboarding]);

  // Splash while we check AsyncStorage (usually < 100 ms)
  if (isLoading) {
    return (
      <View style={{ flex: 1, alignItems: "center", justifyContent: "center", backgroundColor: "#F5F2EE" }}>
        <Image
          source={images.appLogo}
          style={{ width: 180, height: 180, marginBottom: 24 }}
          resizeMode="contain"
        />
        <ActivityIndicator size="large" color="#7D6B55" />
      </View>
    );
  }

  return <Stack screenOptions={{ headerShown: false }} />;
}

// ── Root layout: wraps everything in context providers ────────────────────────
export default function RootLayout() {
  return (
    <UserProvider>
      <AuthProvider>
        <LawyerProvider>
          <RootNavigator />
        </LawyerProvider>
      </AuthProvider>
    </UserProvider>
  );
}
