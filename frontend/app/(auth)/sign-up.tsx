import { Text, View, Image, TextInput, TouchableOpacity, KeyboardAvoidingView, Platform, ScrollView, TouchableWithoutFeedback, Keyboard } from "react-native";
import { useState } from "react";
import images from "@/constants/images";
import { Ionicons } from "@expo/vector-icons";

export default function SignUp() {
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);

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
          <View className="items-center mb-10 w-full">
            <Image source={images.logo} className="img-logo-2 mb-2" resizeMode="contain" />
            <Text className="text-title text-black text-lg mt-4">Sing Up</Text>
          </View>
          
          <View className="w-[85%] gap-4">
            <Text className="text-base text-primary font-inter-semibold">Username:</Text>
            <View className="bg-white rounded-full px-6 h-14 justify-center shadow-sm flex-row items-center gap-2">
              <Ionicons name="person-circle-outline" size={22} color="#B0B0B0" />
              <TextInput 
                placeholder="Enter your username" 
                placeholderTextColor="#B0B0B0" 
                className="flex-1 text-base text-black"
                autoCapitalize="none"
              />
            </View>

            <Text className="text-base text-primary font-inter-semibold">Email or Phone no:</Text> 
            <View className="bg-white rounded-full px-6 h-14 justify-center shadow-sm flex-row items-center gap-2">
              <Ionicons name="mail-outline" size={22} color="#B0B0B0" />
              <TextInput 
                placeholder="Enter your email or phone number" 
                placeholderTextColor="#B0B0B0" 
                className="flex-1 text-base text-black"
                keyboardType="email-address"
                autoCapitalize="none"
              />
            </View>

            <Text className="text-base text-primary font-inter-semibold">Password:</Text>
            <View className="bg-white rounded-full px-6 h-14 justify-center shadow-sm flex-row items-center gap-2">
              <Ionicons name="lock-open-outline" size={22} color="#B0B0B0" />
              <TextInput 
                placeholder="Password" 
                placeholderTextColor="#B0B0B0" 
                secureTextEntry={true}
                className="flex-1 text-base text-black"
              />
              <TouchableOpacity onPress={() => setShowConfirmPassword(!showConfirmPassword)}>
                <Ionicons name={showConfirmPassword ? "eye-off" : "eye"} size={22} color="#B0B0B0" />
              </TouchableOpacity>
            </View>

            <Text className="text-base text-primary font-inter-semibold">Confirm Password:</Text>
            <View className="bg-white rounded-full px-6 h-14 justify-center shadow-sm flex-row items-center gap-2">
              <Ionicons name="lock-open-outline" size={22} color="#B0B0B0" />
              <TextInput 
                placeholder="Confirm Password" 
                placeholderTextColor="#B0B0B0" 
                secureTextEntry={!showConfirmPassword}
                className="flex-1 text-base text-black"
              />
              <TouchableOpacity onPress={() => setShowConfirmPassword(!showConfirmPassword)}>
                <Ionicons name={showConfirmPassword ? "eye-off" : "eye"} size={22} color="#B0B0B0" />
              </TouchableOpacity>
            </View>
            
            <View className="mt-8 items-center w-full">
              <TouchableOpacity className="btn-primary w-full items-center justify-center h-14 rounded-full">
                <Text className="text-button text-white text-base">Sign up</Text>
              </TouchableOpacity>
            </View>
          </View>  
        </ScrollView>
      </TouchableWithoutFeedback>
    </KeyboardAvoidingView>
  );
}
