import "@/global.css"
import { Image, Text, View, TouchableOpacity } from "react-native";
import { router } from "expo-router";
import images from "@/constants/images";
 
export default function App() {
  return (
    <View className="bg-call items-center justify-center">
      <Image source={images.logo} className="img-logo mb-20" />
      <TouchableOpacity className="btn-primary mt-30 w-3/4 items-center justify-center" onPress={() => router.replace('/home')}>
        <Text className="text-button text-lg">Get Started</Text>
      </TouchableOpacity>
    </View>
  );
}