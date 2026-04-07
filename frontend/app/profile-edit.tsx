import { useState } from "react";
import {
  View, Text, ScrollView, TouchableOpacity, TextInput,
  Image, Platform, Alert,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { router } from "expo-router";
import * as ImagePicker from "expo-image-picker";
import { useUser, useTheme, useTranslation } from "@/context/UserContext";

export default function ProfileEdit() {
  const { user, updateUser } = useUser();
  const theme = useTheme();
  const { t, isRTL } = useTranslation();

  const [name, setName] = useState(user.name);
  const [email, setEmail] = useState(user.email);
  const [phone, setPhone] = useState(user.phone);
  const [city, setCity] = useState(user.city);
  const [avatarUri, setAvatarUri] = useState(user.avatarUri);
  const [saving, setSaving] = useState(false);

  const pickAvatar = async () => {
    Alert.alert("Photo de profil", "Choisissez une source", [
      {
        text: "Caméra", onPress: async () => {
          const { status } = await ImagePicker.requestCameraPermissionsAsync();
          if (status !== "granted") { Alert.alert("Permission refusée"); return; }
          const result = await ImagePicker.launchCameraAsync({ quality: 0.8, allowsEditing: true, aspect: [1, 1] });
          if (!result.canceled) setAvatarUri(result.assets[0].uri);
        }
      },
      {
        text: "Galerie", onPress: async () => {
          const result = await ImagePicker.launchImageLibraryAsync({ quality: 0.8, allowsEditing: true, aspect: [1, 1] });
          if (!result.canceled) setAvatarUri(result.assets[0].uri);
        }
      },
      { text: t("cancel"), style: "cancel" },
    ]);
  };

  const handleSave = async () => {
    if (!name.trim()) { Alert.alert("Erreur", t("nameRequired")); return; }
    setSaving(true);
    await updateUser({ name: name.trim(), email: email.trim(), phone: phone.trim(), city: city.trim(), avatarUri });
    setSaving(false);
    Alert.alert("✓", t("savedSuccess"), [{ text: "OK", onPress: () => router.back() }]);
  };

  const fields = [
    { label: t("fullName"), value: name, setter: setName, icon: "person-outline", placeholder: t("namePlaceholder"), keyboard: "default" },
    { label: t("emailAddress"), value: email, setter: setEmail, icon: "mail-outline", placeholder: t("emailPlaceholder"), keyboard: "email-address" },
    { label: t("phone"), value: phone, setter: setPhone, icon: "call-outline", placeholder: t("phonePlaceholder"), keyboard: "phone-pad" },
    { label: t("city"), value: city, setter: setCity, icon: "location-outline", placeholder: t("cityPlaceholder"), keyboard: "default" },
  ];

  return (
    <View style={{ flex: 1, backgroundColor: theme.background }}>
      {/* Header */}
      <View style={{
        paddingTop: Platform.OS === "ios" ? 56 : 44,
        paddingBottom: 20, paddingHorizontal: 20,
        backgroundColor: theme.headerBg,
        borderBottomLeftRadius: 28, borderBottomRightRadius: 28,
        flexDirection: isRTL ? "row-reverse" : "row",
        alignItems: "center", gap: 14,
      }}>
        <TouchableOpacity onPress={() => router.back()} style={{
          width: 40, height: 40, borderRadius: 12,
          backgroundColor: "rgba(255,255,255,0.3)",
          alignItems: "center", justifyContent: "center",
        }}>
          <Ionicons name={isRTL ? "arrow-forward" : "arrow-back"} size={22} color={theme.text} />
        </TouchableOpacity>
        <Text style={{ fontSize: 20, fontFamily: "inter-semibold", color: theme.text, flex: 1, textAlign: isRTL ? "right" : "left" }}>
          {t("profileTitle")}
        </Text>
        <TouchableOpacity onPress={handleSave} disabled={saving} style={{
          backgroundColor: theme.primary, borderRadius: 12,
          paddingHorizontal: 16, paddingVertical: 8,
        }}>
          <Text style={{ fontSize: 13, fontFamily: "inter-semibold", color: "#fff" }}>
            {saving ? t("saving") : t("save")}
          </Text>
        </TouchableOpacity>
      </View>

      <ScrollView contentContainerStyle={{ padding: 20, gap: 20 }} showsVerticalScrollIndicator={false}>
        {/* Avatar */}
        <View style={{ alignItems: "center", marginVertical: 10 }}>
          <TouchableOpacity onPress={pickAvatar} activeOpacity={0.85}>
            <View style={{
              width: 100, height: 100, borderRadius: 50,
              backgroundColor: theme.primary,
              alignItems: "center", justifyContent: "center",
              borderWidth: 4, borderColor: theme.card,
              shadowColor: theme.shadow, shadowOffset: { width: 0, height: 4 },
              shadowOpacity: 0.15, shadowRadius: 10, elevation: 6, overflow: "hidden",
            }}>
              {avatarUri
                ? <Image source={{ uri: avatarUri }} style={{ width: 100, height: 100 }} />
                : <Ionicons name="person" size={44} color="#FFFFFF" />
              }
            </View>
            <View style={{
              position: "absolute", bottom: 2, right: 2,
              backgroundColor: theme.primary, borderRadius: 14,
              width: 28, height: 28, alignItems: "center", justifyContent: "center",
              borderWidth: 2, borderColor: theme.card,
            }}>
              <Ionicons name="camera" size={14} color="#fff" />
            </View>
          </TouchableOpacity>
          <Text style={{ fontSize: 13, color: theme.textSecondary, marginTop: 10 }}>
            {t("tapToEdit")}
          </Text>
        </View>

        {/* Fields */}
        <View style={{ backgroundColor: theme.card, borderRadius: 20, padding: 20, gap: 18, shadowColor: theme.shadow, shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.06, elevation: 2 }}>
          {fields.map((field) => (
            <View key={field.label}>
              <Text style={{ fontSize: 12, fontFamily: "inter-semibold", color: theme.textSecondary, marginBottom: 8, textTransform: "uppercase", letterSpacing: 0.5, textAlign: isRTL ? "right" : "left" }}>
                {field.label}
              </Text>
              <View style={{ flexDirection: isRTL ? "row-reverse" : "row", alignItems: "center", backgroundColor: theme.inputBg, borderRadius: 14, paddingHorizontal: 14, height: 50, borderWidth: 1, borderColor: theme.border }}>
                <Ionicons name={field.icon as any} size={18} color={theme.textMuted} style={{ marginRight: isRTL ? 0 : 10, marginLeft: isRTL ? 10 : 0 }} />
                <TextInput
                  value={field.value}
                  onChangeText={field.setter}
                  placeholder={field.placeholder}
                  placeholderTextColor={theme.textMuted}
                  keyboardType={field.keyboard as any}
                  autoCapitalize="none"
                  style={{ flex: 1, fontSize: 15, color: theme.text, textAlign: isRTL ? "right" : "left" }}
                />
              </View>
            </View>
          ))}
        </View>

        {/* Privacy note */}
        <View style={{ backgroundColor: theme.primaryLight, borderRadius: 14, padding: 14, flexDirection: isRTL ? "row-reverse" : "row", gap: 10, alignItems: "flex-start", borderWidth: 1, borderColor: theme.primaryMedium }}>
          <Ionicons name="information-circle-outline" size={18} color={theme.primary} style={{ marginTop: 1 }} />
          <Text style={{ flex: 1, fontSize: 12, color: theme.primary, lineHeight: 18, textAlign: isRTL ? "right" : "left" }}>
            {t("privacyNote")}
          </Text>
        </View>
      </ScrollView>
    </View>
  );
}
