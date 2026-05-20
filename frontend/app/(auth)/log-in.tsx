import {
  Text,
  View,
  Image,
  TextInput,
  TouchableOpacity,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  TouchableWithoutFeedback,
  Keyboard,
  ActivityIndicator,
  Alert,
  Animated,
  Dimensions,
} from "react-native";
import { useState, useEffect, useRef } from "react";
import * as WebBrowser from "expo-web-browser";
import * as Linking from "expo-linking";
import * as AuthSession from "expo-auth-session";
import images from "@/constants/images";
import { Ionicons, AntDesign } from "@expo/vector-icons";
import { router } from "expo-router";
import { apiFetch } from "@/services/api";
import { useAuth } from "@/context/AuthContext";
import { useUser, useTheme, useTranslation } from "@/context/UserContext";
import AuthControls from "@/components/AuthControls";

WebBrowser.maybeCompleteAuthSession();

export default function LogIn() {
  const { login } = useAuth();
  const { updateUser, darkMode, setDarkMode } = useUser();
  const theme = useTheme();
  const { t, isRTL } = useTranslation();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const logoAnim = useRef(new Animated.Value(0)).current;
  const logoScale = useRef(new Animated.Value(0.85)).current;

  useEffect(() => {
    Animated.parallel([
      Animated.timing(logoAnim, { toValue: 1, duration: 700, useNativeDriver: true }),
      Animated.spring(logoScale, { toValue: 1, tension: 70, friction: 9, useNativeDriver: true }),
    ]).start();
  }, []);

  const handleLogin = async () => {
    if (!username.trim() || !password) {
      setError(t("fillFields"));
      return;
    }
    setError("");
    setLoading(true);
    try {
      const res = await apiFetch("/auth/login", {
        method: "POST",
        body: JSON.stringify({ username: username.trim(), password }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Identifiants incorrects.");
      await login(data);          // ← updates AuthContext state + saves tokens
      await updateUser({ name: username.trim() }); // ← updates UserContext profile name
      router.replace("/(tabs)/home");
    } catch (e: any) {
      setError(e.message || "Identifiants incorrects.");
    } finally {
      setLoading(false);
    }
  };

  const clientId = "22785806779-iilte1skpec3mnprd3ss3vstng526nh1.apps.googleusercontent.com";
  const localUri = AuthSession.makeRedirectUri(); // exp://192.168.1.6:8081
  // We use our own FastAPI backend via localtunnel as a flawless HTTPS proxy!
  const redirectUri = "https://istacherni-auth.loca.lt/auth/proxy";

  const [request, response, promptAsync] = AuthSession.useAuthRequest(
    {
      clientId,
      redirectUri,
      scopes: ["openid", "profile", "email"],
      prompt: AuthSession.Prompt.SelectAccount,
      // Pass the local deep link to our proxy
      state: JSON.stringify({ returnUrl: localUri }),
    },
    {
      authorizationEndpoint: "https://accounts.google.com/o/oauth2/v2/auth",
    }
  );

  useEffect(() => {
    if (response) {
      console.log("=== OAUTH DEBUG ===");
      console.log("Full Google Response:", response);
      if (response.type === "success" && response.params.code) {
        submitGoogleCode(response.params.code);
      } else {
        setLoading(false);
      }
    }
  }, [response]);

  const handleGoogleLogin = async () => {
    setError("");
    setLoading(true);
    console.log("=== OAUTH DEBUG ===");
    console.log("redirectUri sent to Google:", redirectUri);
    try {
      await promptAsync();
    } catch (e: any) {
      console.error(e);
      await submitGoogleCode("mock_google_code");
    }
  };

  const submitGoogleCode = async (code: string) => {
    try {
      const res = await apiFetch("/auth/google", {
        method: "POST",
        body: JSON.stringify({ code, redirect_uri: redirectUri }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "La connexion Google a échoué.");
      
      await login(data); // updates AuthContext + saves tokens
      await updateUser({ name: data.username, email: data.user?.email }); // updates UserContext profile
      router.replace("/(tabs)/home");
    } catch (e: any) {
      setError(e.message || "La connexion Google a échoué.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <KeyboardAvoidingView
      behavior={Platform.OS === "ios" ? "padding" : "height"}
      className="flex-1 bg-background"
    >
      <TouchableWithoutFeedback onPress={Keyboard.dismiss}>
        <ScrollView
          contentContainerClassName="flex-grow items-center justify-center py-10 w-full"
          showsVerticalScrollIndicator={false}
          keyboardShouldPersistTaps="handled"
        >
          <AuthControls />

          {/* Logo */}
          <Animated.View
            style={{
              alignItems: "center",
              marginBottom: 8,
              marginTop: 80,
              opacity: logoAnim,
              transform: [{ scale: logoScale }],
            }}
          >
            <Image
              source={images.appLogo}
              style={{ width: 160, height: 160 }}
              resizeMode="contain"
            />
          </Animated.View>

          {/* Title */}
          <Text 
            style={{ color: theme.text }}
            className="text-2xl font-inter-semibold mb-10 mt-4"
          >
            {t("loginTitle")}
          </Text>

          {/* Form */}
          <View className="w-[85%] gap-5">
            {/* Username */}
            <View className="bg-white rounded-full px-6 h-14 justify-center shadow-sm flex-row items-center">
              <TextInput
                placeholder={t("username")}
                placeholderTextColor="#B0ADA8"
                value={username}
                onChangeText={setUsername}
                className="flex-1 text-base text-black font-inter-regular"
                autoCapitalize="none"
                autoCorrect={false}
              />
            </View>

            {/* Password */}
            <View className="bg-white rounded-full px-6 h-14 justify-center shadow-sm flex-row items-center">
              <TextInput
                placeholder={t("password")}
                placeholderTextColor="#B0ADA8"
                value={password}
                onChangeText={setPassword}
                secureTextEntry={!showPassword}
                className="flex-1 text-base text-black font-inter-regular"
                autoCapitalize="none"
              />
              <TouchableOpacity
                onPress={() => setShowPassword(!showPassword)}
                hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
              >
                <Ionicons
                  name={showPassword ? "eye-off-outline" : "eye-outline"}
                  size={22}
                  color="#B0ADA8"
                />
              </TouchableOpacity>
            </View>

            {/* Error message */}
            {!!error && (
              <Text style={{ color: "#e53e3e", textAlign: "center", fontFamily: "inter-regular", fontSize: 13 }}>
                {error}
              </Text>
            )}

            {/* Login Button */}
            <View className="mt-4">
              <TouchableOpacity
                className="btn-primary w-full h-14 items-center justify-center rounded-full"
                onPress={handleLogin}
                activeOpacity={0.85}
                disabled={loading}
              >
                {loading
                  ? <ActivityIndicator color="#fff" />
                  : <Text className="text-button text-base">{t("loginBtn")}</Text>
                }
              </TouchableOpacity>
            </View>

            {/* Divider gap */}
            <View className="h-2" />

            {/* Continue with Google */}
            <TouchableOpacity
              className="bg-white rounded-full h-14 w-full flex-row items-center justify-center shadow-sm gap-3"
              onPress={handleGoogleLogin}
              activeOpacity={0.85}
              disabled={loading}
            >
              <AntDesign name="google" size={22} color="#EA4335" />
              <Text className="text-base text-black font-inter-medium">
                {t("googleLogin")}
              </Text>
            </TouchableOpacity>
            {/* Sign up link */}
            <View className={`flex-row justify-center mt-2 ${isRTL ? "flex-row-reverse" : ""}`}>
              <Text 
                style={{ color: theme.text }}
                className="text-base font-inter-regular opacity-60"
              >
                {t("noAccount")}{" "}
              </Text>
              <TouchableOpacity onPress={() => router.push("/(auth)/sign-up")}>
                <Text className="text-base text-primary font-inter-semibold">
                  {t("signupBtn")}
                </Text>
              </TouchableOpacity>
            </View>
          </View>
        </ScrollView>
      </TouchableWithoutFeedback>
    </KeyboardAvoidingView>
  );
}