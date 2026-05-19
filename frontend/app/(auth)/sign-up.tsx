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

export default function SignUp() {
  const { login } = useAuth();
  const { updateUser, darkMode, setDarkMode } = useUser();
  const theme = useTheme();
  const { t, isRTL } = useTranslation();
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
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

  const handleSignUp = async () => {
    if (!username.trim() || !email.trim() || !password || !confirmPassword) {
      setError(t("fillFields"));
      return;
    }
    if (password !== confirmPassword) {
      setError("Les mots de passe ne correspondent pas.");
      return;
    }
    if (password.length < 8) {
      setError("Le mot de passe doit contenir au moins 8 caractères.");
      return;
    }
    setError("");
    setLoading(true);
    try {
      // 1. Register
      const regRes = await apiFetch("/auth/register", {
        method: "POST",
        body: JSON.stringify({ username: username.trim(), email: email.trim(), password }),
      });
      const regData = await regRes.json();
      if (!regRes.ok) throw new Error(regData.detail || "L'inscription a échoué.");

      // 2. Auto-login to get tokens
      const loginRes = await apiFetch("/auth/login", {
        method: "POST",
        body: JSON.stringify({ username: username.trim(), password }),
      });
      const loginData = await loginRes.json();
      if (!loginRes.ok) throw new Error(loginData.detail || "Connexion automatique échouée.");

      await login(loginData);       // ← updates AuthContext + saves tokens
      await updateUser({ name: username.trim(), email: email.trim() }); // ← updates UserContext profile
      router.replace("/(tabs)/profile");
    } catch (e: any) {
      setError(e.message || "L'inscription a échoué.");
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
              style={{ width: 140, height: 140 }}
              resizeMode="contain"
            />
          </Animated.View>

          {/* Title */}
          <Text 
            style={{ color: theme.text }}
            className="text-2xl font-inter-semibold mb-10 mt-4"
          >
            {t("signupTitle")}
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

            {/* Email or Phone */}
            <View className="bg-white rounded-full px-6 h-14 justify-center shadow-sm flex-row items-center">
              <TextInput
                placeholder={t("emailPhone")}
                placeholderTextColor="#B0ADA8"
                value={email}
                onChangeText={setEmail}
                className="flex-1 text-base text-black font-inter-regular"
                keyboardType="email-address"
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
                secureTextEntry={true}
                className="flex-1 text-base text-black font-inter-regular"
                autoCapitalize="none"
              />
            </View>

            {/* Confirm Password */}
            <View className="bg-white rounded-full px-6 h-14 justify-center shadow-sm flex-row items-center">
              <TextInput
                placeholder={t("confirmPassword")}
                placeholderTextColor="#B0ADA8"
                value={confirmPassword}
                onChangeText={setConfirmPassword}
                secureTextEntry={!showConfirmPassword}
                className="flex-1 text-base text-black font-inter-regular"
                autoCapitalize="none"
              />
              <TouchableOpacity
                onPress={() => setShowConfirmPassword(!showConfirmPassword)}
                hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
              >
                <Ionicons
                  name={showConfirmPassword ? "eye-off-outline" : "eye-outline"}
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

            {/* Sign Up Button */}
            <View className="mt-4">
              <TouchableOpacity
                className="btn-primary w-full h-14 items-center justify-center rounded-full"
                onPress={handleSignUp}
                activeOpacity={0.85}
                disabled={loading}
              >
                {loading
                  ? <ActivityIndicator color="#fff" />
                  : <Text className="text-button text-base">{t("signupBtn")}</Text>
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

            {/* Login link */}
            <View className={`flex-row justify-center mt-2 ${isRTL ? "flex-row-reverse" : ""}`}>
              <Text 
                style={{ color: theme.text }}
                className="text-base font-inter-regular opacity-60"
              >
                {t("alreadyHaveAccount")}{" "}
              </Text>
              <TouchableOpacity onPress={() => router.push("/(auth)/log-in")}>
                <Text className="text-base text-primary font-inter-semibold">
                  {t("loginBtn")}
                </Text>
              </TouchableOpacity>
            </View>
          </View>
        </ScrollView>
      </TouchableWithoutFeedback>
    </KeyboardAvoidingView>
  );
}
