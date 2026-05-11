import { useEffect } from "react";
import { ActivityIndicator, View } from "react-native";
import { router, Stack } from "expo-router";
import "@/global.css";
import { UserProvider } from "@/context/UserContext";
import { AuthProvider, useAuth } from "@/context/AuthContext";
import { LawyerProvider } from "@/context/LawyerContext";

// ── Inner layout: reads auth state and enforces the guard ─────────────────────
function RootNavigator() {
  const { isAuthenticated, isLoading } = useAuth();

  useEffect(() => {
    if (isLoading) return; // wait until AsyncStorage check is done

    if (isAuthenticated) {
      // Already logged in — go straight to the app
      router.replace("/(tabs)/home");
    } else {
      // No valid token — force login
      router.replace("/(auth)/log-in");
    }
  }, [isAuthenticated, isLoading]);

  // Splash while we check AsyncStorage (usually < 100 ms)
  if (isLoading) {
    return (
      <View style={{ flex: 1, alignItems: "center", justifyContent: "center", backgroundColor: "#fff" }}>
        <ActivityIndicator size="large" color="#4B7BEC" />
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
