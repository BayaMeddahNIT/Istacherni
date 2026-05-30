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
  Animated,
} from "react-native";
import { useState, useEffect, useRef } from "react";
import images from "@/constants/images";
import { Ionicons } from "@expo/vector-icons";
import { router } from "expo-router";
import { apiFetch } from "@/services/api";
import { useAuth } from "@/context/AuthContext";
import { useUser, useTheme, useTranslation } from "@/context/UserContext";
import AuthControls from "@/components/AuthControls";

export default function SignUp() {
  const { login } = useAuth();
  const { updateUser } = useUser();
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

  return (
    <KeyboardAvoidingView
      behavior={Platform.OS === "ios" ? "padding" : "height"}
      style={{ backgroundColor: theme.background }}
      className="flex-1"
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
            <View 
              style={{ backgroundColor: theme.card }}
              className="rounded-full px-6 h-14 justify-center shadow-sm flex-row items-center"
            >
              <TextInput
                placeholder={t("username")}
                placeholderTextColor={theme.textMuted}
                value={username}
                onChangeText={setUsername}
                style={{
                  flex: 1,
                  fontSize: 16,
                  color: theme.text,
                  fontFamily: "inter-regular",
                  textAlign: isRTL ? "right" : "left",
                }}
                autoCapitalize="none"
                autoCorrect={false}
              />
            </View>

            {/* Email or Phone */}
            <View 
              style={{ backgroundColor: theme.card }}
              className="rounded-full px-6 h-14 justify-center shadow-sm flex-row items-center"
            >
              <TextInput
                placeholder={t("emailPhone")}
                placeholderTextColor={theme.textMuted}
                value={email}
                onChangeText={setEmail}
                style={{
                  flex: 1,
                  fontSize: 16,
                  color: theme.text,
                  fontFamily: "inter-regular",
                  textAlign: isRTL ? "right" : "left",
                }}
                keyboardType="email-address"
                autoCapitalize="none"
                autoCorrect={false}
              />
            </View>

            {/* Password */}
            <View 
              style={{ backgroundColor: theme.card }}
              className="rounded-full px-6 h-14 justify-center shadow-sm flex-row items-center"
            >
              <TextInput
                placeholder={t("password")}
                placeholderTextColor={theme.textMuted}
                value={password}
                onChangeText={setPassword}
                secureTextEntry={true}
                style={{
                  flex: 1,
                  fontSize: 16,
                  color: theme.text,
                  textAlign: isRTL ? "right" : "left",
                }}
                autoCapitalize="none"
              />
            </View>

            {/* Confirm Password */}
            <View 
              style={{ backgroundColor: theme.card }}
              className="rounded-full px-6 h-14 justify-center shadow-sm flex-row items-center"
            >
              <TextInput
                placeholder={t("confirmPassword")}
                placeholderTextColor={theme.textMuted}
                value={confirmPassword}
                onChangeText={setConfirmPassword}
                secureTextEntry={!showConfirmPassword}
                style={{
                  flex: 1,
                  fontSize: 16,
                  color: theme.text,
                  textAlign: isRTL ? "right" : "left",
                }}
                autoCapitalize="none"
              />
              <TouchableOpacity
                onPress={() => setShowConfirmPassword(!showConfirmPassword)}
                hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
              >
                <Ionicons
                  name={showConfirmPassword ? "eye-off-outline" : "eye-outline"}
                  size={22}
                  color={theme.textMuted}
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
                style={{ backgroundColor: theme.primary }}
                className="w-full h-14 items-center justify-center rounded-full"
                onPress={handleSignUp}
                activeOpacity={0.85}
                disabled={loading}
              >
                {loading
                  ? <ActivityIndicator color="#fff" />
                  : <Text style={{ color: "#fff" }} className="font-inter-semibold text-base">{t("signupBtn")}</Text>
                }
              </TouchableOpacity>
            </View>

            {/* Login link */}
            <View className={`flex-row justify-center mt-2 ${isRTL ? "flex-row-reverse" : ""}`}>
              <Text 
                style={{ color: theme.text }}
                className="text-base font-inter-regular opacity-60"
              >
                {t("alreadyHaveAccount")}{" "}
              </Text>
              <TouchableOpacity onPress={() => router.push("/(auth)/log-in")}>
                <Text style={{ color: theme.primary }} className="text-base font-inter-semibold">
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
