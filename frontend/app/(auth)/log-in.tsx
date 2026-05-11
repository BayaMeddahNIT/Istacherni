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
} from "react-native";
import { useState, useEffect } from "react";
import * as WebBrowser from "expo-web-browser";
import * as Linking from "expo-linking";
import * as AuthSession from "expo-auth-session";
import images from "@/constants/images";
import { Ionicons, AntDesign } from "@expo/vector-icons";
import { router } from "expo-router";
import { apiFetch } from "@/services/api";
import { useAuth } from "@/context/AuthContext";
import { useUser, useTheme, useTranslation } from "@/context/UserContext";
import LanguageSelector from "@/components/LanguageSelector";

WebBrowser.maybeCompleteAuthSession();

export default function LogIn() {
  const { login } = useAuth();
  const { updateUser } = useUser();
  const theme = useTheme();
  const { t, isRTL } = useTranslation();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

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
          {/* Language Selector */}
          <View style={{
            position: "absolute",
            top: Platform.OS === 'ios' ? 40 : 20,
            right: isRTL ? undefined : 20,
            left: isRTL ? 20 : undefined,
            zIndex: 1000
          }}>
            <LanguageSelector />
          </View>

          {/* Logo */}
          <View className="items-center mb-2 w-full">
            <Image
              source={images.logo}
              className="img-logo-2"
              resizeMode="contain"
            />
          </View>

          {/* Title */}
          <Text className="text-2xl font-inter-semibold text-black mb-10 mt-4">
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
              <Text className="text-base text-black font-inter-regular opacity-60">
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