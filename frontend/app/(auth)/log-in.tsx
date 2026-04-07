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
} from "react-native";
import { useState } from "react";
import images from "@/constants/images";
import { Ionicons, AntDesign } from "@expo/vector-icons";
import { router } from "expo-router";

export default function LogIn() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);

  const handleLogin = () => {
    // TODO: implement login logic
    console.log("Login pressed", { username, password });
  };

  const handleGmail = () => {
    // TODO: implement Google OAuth
    console.log("Continue with Gmail pressed");
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
            Se Connecter
          </Text>

          {/* Form */}
          <View className="w-[85%] gap-5">
            {/* Username */}
            <View className="bg-white rounded-full px-6 h-14 justify-center shadow-sm flex-row items-center">
              <TextInput
                placeholder="Username"
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
                placeholder="Password"
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

            {/* Login Button */}
            <View className="mt-4">
              <TouchableOpacity
                className="btn-primary w-full h-14 items-center justify-center rounded-full"
                onPress={handleLogin}
                activeOpacity={0.85}
              >
                <Text className="text-button text-base">Login</Text>
              </TouchableOpacity>
            </View>

            {/* Divider gap */}
            <View className="h-2" />

            {/* Continue with Gmail */}
            <TouchableOpacity
              className="bg-white rounded-full h-14 w-full flex-row items-center justify-center shadow-sm gap-3"
              onPress={handleGmail}
              activeOpacity={0.85}
            >
              <AntDesign name="google" size={22} color="#EA4335" />
              <Text className="text-base text-black font-inter-medium">
                Continue with Gmail
              </Text>
            </TouchableOpacity>
            {/* Sign up link */}
            <View className="flex-row justify-center mt-2">
              <Text className="text-base text-black font-inter-regular opacity-60">
                Pas encore de compte?{" "}
              </Text>
              <TouchableOpacity onPress={() => router.push("/(auth)/sign-up")}>
                <Text className="text-base text-primary font-inter-semibold">
                  S'inscrire
                </Text>
              </TouchableOpacity>
            </View>
          </View>
        </ScrollView>
      </TouchableWithoutFeedback>
    </KeyboardAvoidingView>
  );
}