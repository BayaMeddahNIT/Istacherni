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
import { Ionicons } from "@expo/vector-icons";
import { router } from "expo-router";

export default function SignUp() {
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);

  const handleSignUp = () => {
    // TODO: implement sign-up logic
    console.log("Sign up pressed", { username, email, password, confirmPassword });
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
            S'inscrire
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

            {/* Email or Phone */}
            <View className="bg-white rounded-full px-6 h-14 justify-center shadow-sm flex-row items-center">
              <TextInput
                placeholder="Email or Phone no."
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
                placeholder="Password"
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
                placeholder="Confirm Password"
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

            {/* Sign Up Button */}
            <View className="mt-4">
              <TouchableOpacity
                className="btn-primary w-full h-14 items-center justify-center rounded-full"
                onPress={handleSignUp}
                activeOpacity={0.85}
              >
                <Text className="text-button text-base">Sign up</Text>
              </TouchableOpacity>
            </View>

            {/* Login link */}
            <View className="flex-row justify-center mt-2">
              <Text className="text-base text-black font-inter-regular opacity-60">
                Déjà inscrit?{" "}
              </Text>
              <TouchableOpacity onPress={() => router.push("/(auth)/log-in")}>
                <Text className="text-base text-primary font-inter-semibold">
                  Se connecter
                </Text>
              </TouchableOpacity>
            </View>
          </View>
        </ScrollView>
      </TouchableWithoutFeedback>
    </KeyboardAvoidingView>
  );
}
