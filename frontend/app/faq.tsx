import { useState } from "react";
import { View, Text, ScrollView, TouchableOpacity, Platform, LayoutAnimation, UIManager } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { router } from "expo-router";
import { useTheme, useTranslation } from "@/context/UserContext";

if (Platform.OS === "android" && UIManager.setLayoutAnimationEnabledExperimental) {
  UIManager.setLayoutAnimationEnabledExperimental(true);
}

import translations from "@/constants/i18n";

function AccordionItem({ question, answer }: { question: string; answer: string }) {
  const theme = useTheme();
  const { isRTL } = useTranslation();
  const [open, setOpen] = useState(false);

  return (
    <TouchableOpacity onPress={() => { LayoutAnimation.configureNext(LayoutAnimation.Presets.easeInEaseOut); setOpen(!open); }} activeOpacity={0.85} style={{ borderBottomWidth: 0.5, borderBottomColor: theme.divider, paddingVertical: 14 }}>
      <View style={{ flexDirection: isRTL ? "row-reverse" : "row", alignItems: "flex-start", gap: 12 }}>
        <View style={{ width: 24, height: 24, borderRadius: 7, backgroundColor: open ? theme.primary : theme.pillBg, alignItems: "center", justifyContent: "center", flexShrink: 0, marginTop: 1 }}>
          <Ionicons name={open ? "remove" : "add"} size={14} color={open ? "#fff" : theme.primary} />
        </View>
        <Text style={{ flex: 1, fontSize: 14, fontFamily: "inter-semibold", color: theme.text, lineHeight: 20, textAlign: isRTL ? "right" : "left" }}>
          {question}
        </Text>
      </View>
      {open && (
        <Text style={{ fontSize: 13, color: theme.textSecondary, lineHeight: 20, marginTop: 10, marginLeft: isRTL ? 0 : 36, marginRight: isRTL ? 36 : 0, textAlign: isRTL ? "right" : "left" }}>
          {answer}
        </Text>
      )}
    </TouchableOpacity>
  );
}

export default function FAQ() {
  const theme = useTheme();
  const { t, language, isRTL } = useTranslation();
  const faqData = (translations as any)[language].faqData || [];
  const totalQuestions = faqData.reduce((acc: number, c: any) => acc + c.questions.length, 0);

  return (
    <View style={{ flex: 1, backgroundColor: theme.background }}>
      <View style={{
        paddingTop: Platform.OS === "ios" ? 56 : 44, paddingBottom: 20, paddingHorizontal: 20,
        backgroundColor: theme.headerBg, borderBottomLeftRadius: 28, borderBottomRightRadius: 28,
        flexDirection: isRTL ? "row-reverse" : "row", alignItems: "center", gap: 14,
      }}>
        <TouchableOpacity onPress={() => router.back()} style={{ width: 40, height: 40, borderRadius: 12, backgroundColor: "rgba(255,255,255,0.3)", alignItems: "center", justifyContent: "center" }}>
          <Ionicons name={isRTL ? "arrow-forward" : "arrow-back"} size={22} color={theme.text} />
        </TouchableOpacity>
        <View style={{ flex: 1 }}>
          <Text style={{ fontSize: 20, fontFamily: "inter-semibold", color: theme.text, textAlign: isRTL ? "right" : "left" }}>{t("faqTitle")}</Text>
          <Text style={{ fontSize: 12, color: theme.primary, marginTop: 2, textAlign: isRTL ? "right" : "left" }}>
            {totalQuestions} {t("questionsAnswered")}
          </Text>
        </View>
      </View>

      <ScrollView contentContainerStyle={{ padding: 20, gap: 20 }} showsVerticalScrollIndicator={false}>
        {faqData.map((section: any) => (
          <View key={section.category}>
            <Text style={{ fontSize: 12, fontFamily: "inter-semibold", color: theme.textSecondary, textTransform: "uppercase", letterSpacing: 0.8, marginBottom: 10, marginLeft: 4, textAlign: isRTL ? "right" : "left" }}>
              {section.category}
            </Text>
            <View style={{ backgroundColor: theme.card, borderRadius: 18, paddingHorizontal: 16, paddingVertical: 4, shadowColor: theme.shadow, shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.05, elevation: 2 }}>
              {section.questions.map((item: any, i: number) => (
                <AccordionItem key={i} question={item.q} answer={item.a} />
              ))}
            </View>
          </View>
        ))}

        <View style={{ backgroundColor: theme.primaryLight, borderRadius: 18, padding: 20, alignItems: "center", gap: 12, borderWidth: 1, borderColor: theme.primaryMedium }}>
          <Ionicons name="help-buoy-outline" size={32} color={theme.primary} />
          <Text style={{ fontSize: 15, fontFamily: "inter-semibold", color: theme.text, textAlign: "center" }}>
            {t("stillNeedHelp")}
          </Text>
          <TouchableOpacity onPress={() => router.push("/contact-us")} style={{ backgroundColor: theme.primary, borderRadius: 14, paddingVertical: 12, paddingHorizontal: 24 }}>
            <Text style={{ fontSize: 14, fontFamily: "inter-semibold", color: "#fff" }}>{t("contactUsBtn")}</Text>
          </TouchableOpacity>
        </View>
      </ScrollView>
    </View>
  );
}
