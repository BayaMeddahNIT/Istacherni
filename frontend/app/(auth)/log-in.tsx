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
                secureTextEntry={!showPassword}
                style={{
                  flex: 1,
                  fontSize: 16,
                  color: theme.text,
                  textAlign: isRTL ? "right" : "left",
                }}
                autoCapitalize="none"
              />
              <TouchableOpacity
                onPress={() => setShowPassword(!showPassword)}
                hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
              >
                <Ionicons
                  name={showPassword ? "eye-off-outline" : "eye-outline"}
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

            {/* Login Button */}
            <View className="mt-4">
              <TouchableOpacity
                style={{ backgroundColor: theme.primary }}
                className="w-full h-14 items-center justify-center rounded-full"
                onPress={handleLogin}
                activeOpacity={0.85}
                disabled={loading}
              >
                {loading
                  ? <ActivityIndicator color="#fff" />
                  : <Text style={{ color: "#fff" }} className="font-inter-semibold text-base">{t("loginBtn")}</Text>
                }
              </TouchableOpacity>
            </View>

            {/* Sign up link */}
            <View className={`flex-row justify-center mt-2 ${isRTL ? "flex-row-reverse" : ""}`}>
              <Text 
                style={{ color: theme.text }}
                className="text-base font-inter-regular opacity-60"
              >
                {t("noAccount")}{" "}
              </Text>
              <TouchableOpacity onPress={() => router.push("/(auth)/sign-up")}>
                <Text style={{ color: theme.primary }} className="text-base font-inter-semibold">
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