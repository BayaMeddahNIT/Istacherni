import { useState, useRef } from "react";
import { Image, Text, View, TouchableOpacity, FlatList, Dimensions } from "react-native";
import { router } from "expo-router";
import images from "@/constants/images";
import "@/global.css";

const { width } = Dimensions.get("window");

const onboardingData = [
  {
    id: 1,
    title: "Welcome to Istacherni",
    description: "Find all types of legal services in a single application, with a simple process and multiple benefits.",
    image: images.landing1,
  },
  {
    id: 2,
    title: "Easy Search",
    description: "Enter your city name and the type of consultant you're looking for. Our AI system will select the best candidate for your project.",
    image: images.landing2,
  },
  {
    id: 3,
    title: "Choose the best lawyer",
    description: "Choose the best verified lawyer profiles in your area based on their qualifications, experience, and user reviews.",
    image: images.landing1, //hadi ttbdl
  },
];

export default function Window() {
  const [currentIndex, setCurrentIndex] = useState(0);
  const flatListRef = useRef<FlatList>(null);

  const viewableItemsChanged = useRef(({ viewableItems }: any) => {
    if (viewableItems[0]) {
      setCurrentIndex(viewableItems[0].index);
    }
  }).current;

  const viewConfig = useRef({ viewAreaCoveragePercentThreshold: 50 }).current;

  const handleNext = () => {
    if (currentIndex < onboardingData.length - 1) {
      flatListRef.current?.scrollToIndex({ index: currentIndex + 1, animated: true });
    } else {
      router.push('/(auth)/sign-up');
    }
  };

  return (
    <View className="bg-call">
      {/* Slider Section */}
      <View className="flex-[3] justify-center">
        <FlatList
          data={onboardingData}
          ref={flatListRef}
          horizontal
          pagingEnabled
          showsHorizontalScrollIndicator={false}
          bounces={false}
          keyExtractor={(item) => item.id.toString()}
          onViewableItemsChanged={viewableItemsChanged}
          viewabilityConfig={viewConfig}
          renderItem={({ item }) => (
            <View style={{ width }} className="items-center justify-center p-4">
              <Image source={item.image} className="img-loading " />
              <Text className="text-title">
                {item.title}
              </Text>
              <Text className="text-desc">
                {item.description}
              </Text>
            </View>
          )}
        />
      </View>

      {/* Bottom Section */}
      <View className="flex-[1] items-center px-10 pb-10 justify-end">
        {/* Button */}
        <TouchableOpacity 
          className="btn-primary w-full items-center justify-center mb-10" 
          onPress={handleNext}
        >
          <Text className="text-button text-lg">
            {currentIndex === onboardingData.length - 1 ? "Get Started" : "Suivante >"}
          </Text>
        </TouchableOpacity>

        {/* Progress Points */}
        <View className="flex-row justify-center items-center">
          {onboardingData.map((_, index) => (
            <View
              key={index}
              className={`h-1 mx-2 rounded-full w-8 ${
                currentIndex === index ? "bg-primary" : "bg-primary opacity-30"
              }`}
            />
          ))}
        </View>
      </View>
    </View>
  );
}