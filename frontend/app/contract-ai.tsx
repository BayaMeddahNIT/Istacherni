import { useState, useRef } from "react";
import {
  Text,
  View,
  ScrollView,
  TouchableOpacity,
  TextInput,
  Animated,
  Alert,
  Platform,
  FlatList,
  Modal,
  SafeAreaView,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import * as DocumentPicker from "expo-document-picker";
import * as ImagePicker from "expo-image-picker";
import * as Sharing from "expo-sharing";
import { router } from "expo-router";

// ─── Types ────────────────────────────────────────────────────────────────────

type Tab = "analyze" | "draft" | "history";

interface UploadedFile {
  name: string;
  size?: number;
  type: "pdf" | "image";
  uri: string;
}

type AnalysisStatus = "idle" | "uploading" | "analyzing" | "done";

interface AnalysisResult {
  id: string;
  fileName: string;
  date: string;
  contractType: string;
  parties: string[];
  keyTerms: string[];
  risks: string[];
  recommendations: string[];
  compliance: "✅ Conforme" | "⚠️ À réviser" | "❌ Non conforme";
}

interface HistoryItem {
  id: string;
  fileName: string;
  date: string;
  mode: "analyse" | "brouillon";
  contractType: string;
  compliance?: string;
}

// ─── Mock History ─────────────────────────────────────────────────────────────

const MOCK_HISTORY: HistoryItem[] = [
  {
    id: "h1",
    fileName: "contrat_emploi_2024.pdf",
    date: "04 avril 2025",
    mode: "analyse",
    contractType: "Contrat de Travail",
    compliance: "✅ Conforme",
  },
  {
    id: "h2",
    fileName: "bail_commercial_alger.pdf",
    date: "28 mars 2025",
    mode: "analyse",
    contractType: "Bail Commercial",
    compliance: "⚠️ À réviser",
  },
  {
    id: "h3",
    fileName: "Contrat_Vente_Immo.pdf",
    date: "15 mars 2025",
    mode: "brouillon",
    contractType: "Vente Immobilière",
  },
];

// ─── Contract Types for Drafting ─────────────────────────────────────────────

const CONTRACT_TYPES = [
  { id: "travail", label: "Contrat de Travail", icon: "briefcase-outline", color: "#27AE60" },
  { id: "bail", label: "Bail Commercial", icon: "business-outline", color: "#2980B9" },
  { id: "vente", label: "Vente Immobilière", icon: "home-outline", color: "#E67E22" },
  { id: "service", label: "Prestation de Service", icon: "construct-outline", color: "#8E44AD" },
  { id: "partenariat", label: "Partenariat / JV", icon: "people-outline", color: "#C0392B" },
  { id: "nda", label: "Confidentialité (NDA)", icon: "lock-closed-outline", color: "#34495E" },
];

// ─── Analyze Analysis Progress Step ───────────────────────────────────────────

function ProgressStep({ label, done, active }: { label: string; done: boolean; active: boolean }) {
  return (
    <View style={{ flexDirection: "row", alignItems: "center", marginBottom: 10 }}>
      <View style={{
        width: 28, height: 28, borderRadius: 14,
        backgroundColor: done ? "#27AE60" : active ? "#807261" : "#E0DDD9",
        alignItems: "center", justifyContent: "center", marginRight: 12,
      }}>
        {done
          ? <Ionicons name="checkmark" size={16} color="#fff" />
          : active
            ? <Ionicons name="ellipse" size={10} color="#fff" />
            : <Ionicons name="ellipse-outline" size={14} color="#B0ADA8" />
        }
      </View>
      <Text style={{
        fontSize: 14, fontFamily: done ? "inter-semibold" : "inter-regular",
        color: done ? "#1A1A1A" : active ? "#807261" : "#B0ADA8",
      }}>
        {label}
      </Text>
    </View>
  );
}

// ─── Main Screen ──────────────────────────────────────────────────────────────

export default function ContractAI() {
  const [activeTab, setActiveTab] = useState<Tab>("analyze");

  // — Analyze State —
  const [uploadedFile, setUploadedFile] = useState<UploadedFile | null>(null);
  const [analysisStatus, setAnalysisStatus] = useState<AnalysisStatus>("idle");
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [progressStep, setProgressStep] = useState(0); // 0..3

  // — Draft State —
  const [selectedContract, setSelectedContract] = useState<string | null>(null);
  const [requirements, setRequirements] = useState("");
  const [isDrafting, setIsDrafting] = useState(false);
  const [draftResult, setDraftResult] = useState<string | null>(null);

  // — History —
  const [history, setHistory] = useState<HistoryItem[]>(MOCK_HISTORY);
  const [detailModalItem, setDetailModalItem] = useState<HistoryItem | null>(null);

  // ── File Pickers ─────────────────────────────────────────────────────────

  const pickPDF = async () => {
    try {
      const result = await DocumentPicker.getDocumentAsync({
        type: "application/pdf",
        copyToCacheDirectory: true,
      });
      if (!result.canceled && result.assets?.length) {
        const asset = result.assets[0];
        setUploadedFile({ name: asset.name, size: asset.size, type: "pdf", uri: asset.uri });
        setAnalysisResult(null);
        setAnalysisStatus("idle");
        setProgressStep(0);
      }
    } catch {
      Alert.alert("Erreur", "Impossible d'ouvrir le fichier.");
    }
  };

  const pickImage = async () => {
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    if (status !== "granted") {
      Alert.alert("Permission refusée", "L'accès à la caméra est requis.");
      return;
    }
    Alert.alert(
      "Importer une photo",
      "Choisissez une source",
      [
        {
          text: "Prendre une photo", onPress: async () => {
            const result = await ImagePicker.launchCameraAsync({ quality: 0.85, allowsEditing: true });
            if (!result.canceled) {
              const asset = result.assets[0];
              setUploadedFile({ name: "photo_contrat.jpg", type: "image", uri: asset.uri });
              setAnalysisResult(null); setAnalysisStatus("idle"); setProgressStep(0);
            }
          }
        },
        {
          text: "Galerie", onPress: async () => {
            const result = await ImagePicker.launchImageLibraryAsync({ quality: 0.85, allowsEditing: true });
            if (!result.canceled) {
              const asset = result.assets[0];
              setUploadedFile({ name: "image_contrat.jpg", type: "image", uri: asset.uri });
              setAnalysisResult(null); setAnalysisStatus("idle"); setProgressStep(0);
            }
          }
        },
        { text: "Annuler", style: "cancel" },
      ]
    );
  };

  // ── AI Analysis (simulated) ──────────────────────────────────────────────

  const runAnalysis = async () => {
    if (!uploadedFile) return;
    setAnalysisStatus("uploading");
    setProgressStep(0);

    await delay(1200); setProgressStep(1);    // Lecture du document
    setAnalysisStatus("analyzing");
    await delay(1500); setProgressStep(2);    // Extraction des clauses
    await delay(1500); setProgressStep(3);    // Analyse juridique
    await delay(1200); setProgressStep(4);    // Génération du rapport

    // Simulate result
    setAnalysisResult({
      id: Date.now().toString(),
      fileName: uploadedFile.name,
      date: new Date().toLocaleDateString("fr-DZ"),
      contractType: "Contrat de Travail",
      parties: ["Entreprise SARL AlgérieCode", "M. Mehdi Belkacemi"],
      keyTerms: [
        "Durée: CDI à compter du 1er janvier 2025",
        "Rémunération: 85 000 DZD / mois brut",
        "Période d'essai: 6 mois renouvelable une fois",
        "Préavis: 1 mois pour chaque partie",
        "Non-concurrence: 12 mois post-contrat",
      ],
      risks: [
        "Clause de non-concurrence potentiellement trop large (Loi n° 90-11)",
        "Absence de mention du salaire de base SMIG en cas de révision",
        "Horaires de travail non précisés (Art. 27 Loi 90-11)",
      ],
      recommendations: [
        "Réduire la portée géographique de la clause de non-concurrence",
        "Ajouter une clause d'indexation salariale conforme au SMIG",
        "Préciser les horaires hebdomadaires (44h max selon Art. 27)",
        "Inclure une clause de médiation avant tout recours judiciaire",
      ],
      compliance: "⚠️ À réviser",
    });

    setAnalysisStatus("done");

    // Add to history
    const newItem: HistoryItem = {
      id: Date.now().toString(),
      fileName: uploadedFile.name,
      date: new Date().toLocaleDateString("fr-DZ"),
      mode: "analyse",
      contractType: "Contrat de Travail",
      compliance: "⚠️ À réviser",
    };
    setHistory((prev) => [newItem, ...prev]);
  };

  // ── AI Contract Drafting (simulated) ─────────────────────────────────────

  const runDraft = async () => {
    if (!selectedContract || !requirements.trim()) {
      Alert.alert("Champs requis", "Veuillez sélectionner un type de contrat et saisir vos exigences.");
      return;
    }
    setIsDrafting(true);
    await delay(3500);
    const contractType = CONTRACT_TYPES.find(c => c.id === selectedContract);

    setDraftResult(
      `${contractType?.label?.toUpperCase()}\n` +
      `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n` +
      `ENTRE LES SOUSSIGNÉS :\n\n` +
      `[Partie 1 – à compléter]\n` +
      `SIRET / Registre de Commerce : ______\n\n` +
      `ET :\n\n` +
      `[Partie 2 – à compléter]\n` +
      `Numéro de carte nationale : ______\n\n` +
      `IL A ÉTÉ CONVENU CE QUI SUIT :\n\n` +
      `Article 1 – Objet du Contrat\n` +
      `Le présent contrat a pour objet ${requirements.slice(0, 120)}...\n\n` +
      `Article 2 – Durée\n` +
      `Le présent contrat prend effet à la date de sa signature pour une durée indéterminée / déterminée de [X mois].\n\n` +
      `Article 3 – Obligations des Parties\n` +
      `Chaque partie s'engage à respecter les termes et conditions définis dans le présent accord, conformément au Code Civil algérien (Ord. n° 75-58) et aux lois en vigueur.\n\n` +
      `Article 4 – Rémunération / Contrepartie\n` +
      `En contrepartie des obligations convenues, [Montant / modalités de paiement à préciser].\n\n` +
      `Article 5 – Confidentialité\n` +
      `Les parties s'engagent à garder confidentielles toutes informations échangées dans le cadre de la présente convention.\n\n` +
      `Article 6 – Résiliation\n` +
      `Le présent contrat peut être résilié par l'une ou l'autre des parties moyennant un préavis écrit de [X] jours/mois.\n\n` +
      `Article 7 – Litiges et Juridiction\n` +
      `En cas de litige, les parties conviennent de recourir à la médiation. À défaut d'accord amiable, le différend sera porté devant les tribunaux compétents de la wilaya d'Alger.\n\n` +
      `Fait à Alger, le ________________\n\n` +
      `Signature Partie 1                    Signature Partie 2\n` +
      `____________________         ____________________`
    );
    setIsDrafting(false);

    // Add to history
    const newItem: HistoryItem = {
      id: Date.now().toString(),
      fileName: `brouillon_${contractType?.id}.pdf`,
      date: new Date().toLocaleDateString("fr-DZ"),
      mode: "brouillon",
      contractType: contractType?.label ?? "Contrat",
    };
    setHistory((prev) => [newItem, ...prev]);
  };

  const shareContract = async () => {
    const isAvailable = await Sharing.isAvailableAsync();
    if (isAvailable) {
      Alert.alert("Exporté", "Contrat prêt à être partagé (intégration backend requise).");
    }
  };

  // ─── Render ───────────────────────────────────────────────────────────────

  return (
    <View style={{ flex: 1, backgroundColor: "#DAD6D1" }}>
      {/* Header */}
      <View style={{
        paddingTop: Platform.OS === "ios" ? 56 : 44,
        paddingBottom: 0,
        backgroundColor: "#C8C3BD",
        borderBottomLeftRadius: 0,
        borderBottomRightRadius: 0,
      }}>
        <View style={{ flexDirection: "row", alignItems: "center", paddingHorizontal: 20, paddingBottom: 14 }}>
          <TouchableOpacity onPress={() => router.back()} style={{ marginRight: 12 }}>
            <Ionicons name="arrow-back" size={24} color="#1A1A1A" />
          </TouchableOpacity>
          <View style={{ flex: 1 }}>
            <Text style={{ fontSize: 20, fontFamily: "inter-semibold", color: "#1A1A1A" }}>
              Assistant Juridique IA
            </Text>
            <Text style={{ fontSize: 12, fontFamily: "inter-regular", color: "#807261" }}>
              Analyse & rédaction de contrats
            </Text>
          </View>
          <View style={{
            backgroundColor: "#807261", borderRadius: 10,
            paddingHorizontal: 10, paddingVertical: 4,
          }}>
            <Text style={{ fontSize: 11, color: "#fff", fontFamily: "inter-semibold" }}>IA Juridique</Text>
          </View>
        </View>

        {/* Tab Switcher */}
        <View style={{ flexDirection: "row", paddingHorizontal: 16, gap: 4, paddingBottom: 0 }}>
          {([
            { key: "analyze", label: "Analyser", icon: "analytics-outline" },
            { key: "draft", label: "Rédiger", icon: "create-outline" },
            { key: "history", label: "Historique", icon: "time-outline" },
          ] as const).map((tab) => (
            <TouchableOpacity
              key={tab.key}
              onPress={() => setActiveTab(tab.key)}
              style={{
                flex: 1, flexDirection: "row", alignItems: "center", justifyContent: "center",
                paddingVertical: 12, gap: 5,
                borderBottomWidth: activeTab === tab.key ? 2.5 : 0,
                borderBottomColor: "#807261",
              }}
            >
              <Ionicons name={tab.icon} size={15} color={activeTab === tab.key ? "#807261" : "#B0ADA8"} />
              <Text style={{
                fontSize: 13, fontFamily: activeTab === tab.key ? "inter-semibold" : "inter-regular",
                color: activeTab === tab.key ? "#807261" : "#B0ADA8",
              }}>
                {tab.label}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>

      {/* ── TAB: ANALYZE ─────────────────────────────────────────────────────── */}
      {activeTab === "analyze" && (
        <ScrollView contentContainerStyle={{ padding: 20, gap: 16 }} showsVerticalScrollIndicator={false}>

          {/* Upload Buttons */}
          <View style={{ flexDirection: "row", gap: 12 }}>
            <TouchableOpacity
              onPress={pickPDF}
              activeOpacity={0.85}
              style={{
                flex: 1, backgroundColor: "#FFFFFF", borderRadius: 18, padding: 18,
                alignItems: "center", gap: 10,
                borderWidth: 1.5, borderColor: "#807261", borderStyle: "dashed",
              }}
            >
              <View style={{
                width: 52, height: 52, borderRadius: 14,
                backgroundColor: "#807261" + "18", alignItems: "center", justifyContent: "center",
              }}>
                <Ionicons name="document-attach-outline" size={26} color="#807261" />
              </View>
              <Text style={{ fontSize: 13, fontFamily: "inter-semibold", color: "#807261", textAlign: "center" }}>
                Importer PDF
              </Text>
              <Text style={{ fontSize: 11, fontFamily: "inter-regular", color: "#B0ADA8", textAlign: "center" }}>
                Contrat, accord, bail...
              </Text>
            </TouchableOpacity>

            <TouchableOpacity
              onPress={pickImage}
              activeOpacity={0.85}
              style={{
                flex: 1, backgroundColor: "#FFFFFF", borderRadius: 18, padding: 18,
                alignItems: "center", gap: 10,
                borderWidth: 1.5, borderColor: "#9A8E7F", borderStyle: "dashed",
              }}
            >
              <View style={{
                width: 52, height: 52, borderRadius: 14,
                backgroundColor: "#9A8E7F" + "18", alignItems: "center", justifyContent: "center",
              }}>
                <Ionicons name="camera-outline" size={26} color="#9A8E7F" />
              </View>
              <Text style={{ fontSize: 13, fontFamily: "inter-semibold", color: "#9A8E7F", textAlign: "center" }}>
                Photo
              </Text>
              <Text style={{ fontSize: 11, fontFamily: "inter-regular", color: "#B0ADA8", textAlign: "center" }}>
                Caméra ou galerie
              </Text>
            </TouchableOpacity>
          </View>

          {/* File Preview */}
          {uploadedFile && (
            <View style={{
              backgroundColor: "#FFFFFF", borderRadius: 16, padding: 16,
              flexDirection: "row", alignItems: "center", gap: 14,
              shadowColor: "#000", shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.06, elevation: 2,
            }}>
              <View style={{
                width: 44, height: 52, borderRadius: 10, backgroundColor: "#807261" + "15",
                alignItems: "center", justifyContent: "center",
              }}>
                <Ionicons name={uploadedFile.type === "pdf" ? "document-text" : "image"} size={24} color="#807261" />
              </View>
              <View style={{ flex: 1 }}>
                <Text numberOfLines={1} style={{ fontSize: 14, fontFamily: "inter-semibold", color: "#1A1A1A" }}>
                  {uploadedFile.name}
                </Text>
                {uploadedFile.size && (
                  <Text style={{ fontSize: 12, fontFamily: "inter-regular", color: "#9A8E7F", marginTop: 2 }}>
                    {(uploadedFile.size / 1024).toFixed(1)} Ko
                  </Text>
                )}
                <Text style={{ fontSize: 11, fontFamily: "inter-regular", color: "#27AE60", marginTop: 4 }}>
                  ✓ Fichier prêt pour l'analyse
                </Text>
              </View>
              <TouchableOpacity onPress={() => { setUploadedFile(null); setAnalysisResult(null); setAnalysisStatus("idle"); setProgressStep(0); }}>
                <Ionicons name="close-circle-outline" size={22} color="#B0ADA8" />
              </TouchableOpacity>
            </View>
          )}

          {/* Analyze Button */}
          {uploadedFile && analysisStatus === "idle" && (
            <TouchableOpacity
              onPress={runAnalysis}
              activeOpacity={0.88}
              style={{
                backgroundColor: "#807261", borderRadius: 16, paddingVertical: 18,
                flexDirection: "row", alignItems: "center", justifyContent: "center", gap: 10,
              }}
            >
              <Ionicons name="sparkles" size={20} color="#fff" />
              <Text style={{ fontSize: 16, fontFamily: "inter-semibold", color: "#fff" }}>
                Analyser avec l'IA
              </Text>
            </TouchableOpacity>
          )}

          {/* Progress Indicator */}
          {(analysisStatus === "uploading" || analysisStatus === "analyzing") && (
            <View style={{
              backgroundColor: "#FFFFFF", borderRadius: 18, padding: 22,
              shadowColor: "#000", shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.06, elevation: 2,
            }}>
              <Text style={{ fontSize: 15, fontFamily: "inter-semibold", color: "#1A1A1A", marginBottom: 18 }}>
                Analyse en cours…
              </Text>
              <ProgressStep label="Lecture du document" done={progressStep > 0} active={progressStep === 0} />
              <ProgressStep label="Extraction des clauses" done={progressStep > 1} active={progressStep === 1} />
              <ProgressStep label="Analyse juridique IA" done={progressStep > 2} active={progressStep === 2} />
              <ProgressStep label="Génération du rapport" done={progressStep > 3} active={progressStep === 3} />
            </View>
          )}

          {/* Analysis Result */}
          {analysisStatus === "done" && analysisResult && (
            <>
              {/* Summary Card */}
              <View style={{
                backgroundColor: "#FFFFFF", borderRadius: 18, padding: 20,
                shadowColor: "#000", shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.06, elevation: 2,
              }}>
                <View style={{ flexDirection: "row", alignItems: "center", justifyContent: "space-between", marginBottom: 14 }}>
                  <Text style={{ fontSize: 16, fontFamily: "inter-semibold", color: "#1A1A1A" }}>
                    Rapport d'Analyse
                  </Text>
                  <Text style={{ fontSize: 14, fontFamily: "inter-semibold" }}>
                    {analysisResult.compliance}
                  </Text>
                </View>

                <View style={{ backgroundColor: "#F7F5F3", borderRadius: 12, padding: 14, marginBottom: 14 }}>
                  <Text style={{ fontSize: 12, color: "#9A8E7F", fontFamily: "inter-medium", marginBottom: 4 }}>
                    TYPE DE CONTRAT
                  </Text>
                  <Text style={{ fontSize: 15, fontFamily: "inter-semibold", color: "#1A1A1A" }}>
                    {analysisResult.contractType}
                  </Text>
                  <Text style={{ fontSize: 12, color: "#9A8E7F", fontFamily: "inter-regular", marginTop: 2 }}>
                    Parties: {analysisResult.parties.join(" • ")}
                  </Text>
                </View>

                {/* Key Terms */}
                <Text style={{ fontSize: 13, fontFamily: "inter-semibold", color: "#807261", marginBottom: 8 }}>
                  📋 Clauses Principales
                </Text>
                {analysisResult.keyTerms.map((term, i) => (
                  <View key={i} style={{ flexDirection: "row", alignItems: "flex-start", marginBottom: 6 }}>
                    <Text style={{ color: "#807261", marginRight: 8, fontSize: 13 }}>›</Text>
                    <Text style={{ fontSize: 13, fontFamily: "inter-regular", color: "#1A1A1A", flex: 1 }}>{term}</Text>
                  </View>
                ))}
              </View>

              {/* Risks */}
              <View style={{
                backgroundColor: "#FFF5F5", borderRadius: 18, padding: 20, borderWidth: 1, borderColor: "#FECACA",
              }}>
                <Text style={{ fontSize: 14, fontFamily: "inter-semibold", color: "#C0392B", marginBottom: 12 }}>
                  ⚠️ Risques Identifiés
                </Text>
                {analysisResult.risks.map((risk, i) => (
                  <View key={i} style={{ flexDirection: "row", alignItems: "flex-start", marginBottom: 8 }}>
                    <Ionicons name="alert-circle-outline" size={15} color="#C0392B" style={{ marginRight: 8, marginTop: 1 }} />
                    <Text style={{ fontSize: 13, fontFamily: "inter-regular", color: "#7F1D1D", flex: 1 }}>{risk}</Text>
                  </View>
                ))}
              </View>

              {/* Recommendations */}
              <View style={{
                backgroundColor: "#F0FDF4", borderRadius: 18, padding: 20, borderWidth: 1, borderColor: "#BBF7D0",
              }}>
                <Text style={{ fontSize: 14, fontFamily: "inter-semibold", color: "#15803D", marginBottom: 12 }}>
                  ✅ Recommandations
                </Text>
                {analysisResult.recommendations.map((rec, i) => (
                  <View key={i} style={{ flexDirection: "row", alignItems: "flex-start", marginBottom: 8 }}>
                    <Ionicons name="checkmark-circle-outline" size={15} color="#15803D" style={{ marginRight: 8, marginTop: 1 }} />
                    <Text style={{ fontSize: 13, fontFamily: "inter-regular", color: "#14532D", flex: 1 }}>{rec}</Text>
                  </View>
                ))}
              </View>

              {/* Export */}
              <TouchableOpacity
                onPress={shareContract}
                activeOpacity={0.88}
                style={{
                  backgroundColor: "#F0EDEA", borderRadius: 16, paddingVertical: 16,
                  flexDirection: "row", alignItems: "center", justifyContent: "center", gap: 8,
                }}
              >
                <Ionicons name="share-outline" size={20} color="#807261" />
                <Text style={{ fontSize: 15, fontFamily: "inter-semibold", color: "#807261" }}>
                  Exporter le Rapport
                </Text>
              </TouchableOpacity>
            </>
          )}

          {/* Empty State */}
          {!uploadedFile && analysisStatus === "idle" && (
            <View style={{
              backgroundColor: "#FFFFFF", borderRadius: 18, padding: 32, alignItems: "center",
              shadowColor: "#000", shadowOffset: { width: 0, height: 1 }, shadowOpacity: 0.04, elevation: 1,
            }}>
              <Ionicons name="sparkles-outline" size={48} color="#C8C3BD" />
              <Text style={{ fontSize: 16, fontFamily: "inter-semibold", color: "#9A8E7F", marginTop: 16, textAlign: "center" }}>
                Votre assistant IA est prêt
              </Text>
              <Text style={{ fontSize: 13, fontFamily: "inter-regular", color: "#B0ADA8", marginTop: 8, textAlign: "center", lineHeight: 20 }}>
                Importez un contrat en PDF ou prenez une photo pour commencer l'analyse juridique.
              </Text>
            </View>
          )}
        </ScrollView>
      )}

      {/* ── TAB: DRAFT ───────────────────────────────────────────────────────── */}
      {activeTab === "draft" && (
        <ScrollView contentContainerStyle={{ padding: 20, gap: 16 }} showsVerticalScrollIndicator={false}>

          <Text style={{ fontSize: 14, fontFamily: "inter-semibold", color: "#807261" }}>
            Type de contrat
          </Text>

          {/* Contract Type Grid */}
          <View style={{ flexDirection: "row", flexWrap: "wrap", gap: 10 }}>
            {CONTRACT_TYPES.map((ct) => {
              const active = selectedContract === ct.id;
              return (
                <TouchableOpacity
                  key={ct.id}
                  onPress={() => setSelectedContract(ct.id)}
                  activeOpacity={0.82}
                  style={{
                    width: "47%",
                    backgroundColor: active ? ct.color : "#FFFFFF",
                    borderRadius: 14, padding: 14, gap: 8,
                    borderWidth: active ? 0 : 1.5,
                    borderColor: active ? "transparent" : "#E8E4E0",
                    shadowColor: "#000", shadowOffset: { width: 0, height: 1 },
                    shadowOpacity: active ? 0.15 : 0.04, elevation: active ? 4 : 1,
                  }}
                >
                  <Ionicons name={ct.icon as any} size={22} color={active ? "#fff" : ct.color} />
                  <Text style={{ fontSize: 13, fontFamily: "inter-semibold", color: active ? "#fff" : "#1A1A1A" }}>
                    {ct.label}
                  </Text>
                </TouchableOpacity>
              );
            })}
          </View>

          {/* Requirements Input */}
          <Text style={{ fontSize: 14, fontFamily: "inter-semibold", color: "#807261", marginTop: 4 }}>
            Vos exigences spécifiques
          </Text>
          <View style={{
            backgroundColor: "#FFFFFF", borderRadius: 16, padding: 16,
            shadowColor: "#000", shadowOffset: { width: 0, height: 1 }, shadowOpacity: 0.04, elevation: 1,
          }}>
            <TextInput
              placeholder="Ex: Contrat de travail CDI pour un ingénieur logiciel, salaire 120 000 DZD, période d'essai 3 mois, basé à Alger..."
              placeholderTextColor="#B0ADA8"
              value={requirements}
              onChangeText={setRequirements}
              multiline
              numberOfLines={5}
              style={{
                fontSize: 14, color: "#1A1A1A", fontFamily: "inter-regular",
                textAlignVertical: "top", minHeight: 120,
              }}
            />
          </View>

          {/* Draft Button */}
          <TouchableOpacity
            onPress={runDraft}
            disabled={isDrafting}
            activeOpacity={0.88}
            style={{
              backgroundColor: isDrafting ? "#B0ADA8" : "#807261",
              borderRadius: 16, paddingVertical: 18,
              flexDirection: "row", alignItems: "center", justifyContent: "center", gap: 10,
            }}
          >
            <Ionicons name={isDrafting ? "hourglass-outline" : "create"} size={20} color="#fff" />
            <Text style={{ fontSize: 16, fontFamily: "inter-semibold", color: "#fff" }}>
              {isDrafting ? "Rédaction en cours…" : "Générer avec l'IA"}
            </Text>
          </TouchableOpacity>

          {/* Draft Result */}
          {draftResult && !isDrafting && (
            <>
              <View style={{
                backgroundColor: "#FFFFFF", borderRadius: 18, padding: 20,
                shadowColor: "#000", shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.06, elevation: 2,
              }}>
                <View style={{ flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
                  <Text style={{ fontSize: 15, fontFamily: "inter-semibold", color: "#1A1A1A" }}>
                    Contrat Généré
                  </Text>
                  <View style={{ backgroundColor: "#27AE60" + "20", borderRadius: 8, paddingHorizontal: 8, paddingVertical: 3 }}>
                    <Text style={{ fontSize: 11, color: "#27AE60", fontFamily: "inter-semibold" }}>✓ Brouillon IA</Text>
                  </View>
                </View>
                <ScrollView style={{ maxHeight: 300 }} nestedScrollEnabled showsVerticalScrollIndicator={false}>
                  <Text style={{ fontSize: 12, fontFamily: "inter-regular", color: "#333", lineHeight: 20 }}>
                    {draftResult}
                  </Text>
                </ScrollView>
              </View>

              <View style={{ flexDirection: "row", gap: 10 }}>
                <TouchableOpacity
                  onPress={shareContract}
                  activeOpacity={0.85}
                  style={{
                    flex: 1, backgroundColor: "#807261", borderRadius: 14, paddingVertical: 14,
                    flexDirection: "row", alignItems: "center", justifyContent: "center", gap: 8,
                  }}
                >
                  <Ionicons name="share-outline" size={18} color="#fff" />
                  <Text style={{ fontSize: 14, fontFamily: "inter-semibold", color: "#fff" }}>Exporter</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  onPress={() => { setDraftResult(null); setRequirements(""); setSelectedContract(null); }}
                  activeOpacity={0.85}
                  style={{
                    flex: 1, backgroundColor: "#F0EDEA", borderRadius: 14, paddingVertical: 14,
                    flexDirection: "row", alignItems: "center", justifyContent: "center", gap: 8,
                  }}
                >
                  <Ionicons name="refresh-outline" size={18} color="#807261" />
                  <Text style={{ fontSize: 14, fontFamily: "inter-semibold", color: "#807261" }}>Nouveau</Text>
                </TouchableOpacity>
              </View>
            </>
          )}
        </ScrollView>
      )}

      {/* ── TAB: HISTORY ─────────────────────────────────────────────────────── */}
      {activeTab === "history" && (
        <FlatList
          data={history}
          keyExtractor={(item) => item.id}
          contentContainerStyle={{ padding: 20, gap: 12 }}
          showsVerticalScrollIndicator={false}
          ListEmptyComponent={
            <View style={{ alignItems: "center", paddingTop: 60 }}>
              <Ionicons name="time-outline" size={48} color="#C8C3BD" />
              <Text style={{ fontSize: 15, color: "#9A8E7F", fontFamily: "inter-medium", marginTop: 16 }}>
                Aucun historique
              </Text>
            </View>
          }
          renderItem={({ item }) => (
            <TouchableOpacity
              activeOpacity={0.85}
              style={{
                backgroundColor: "#FFFFFF", borderRadius: 16, padding: 16,
                flexDirection: "row", alignItems: "center", gap: 14,
                shadowColor: "#000", shadowOffset: { width: 0, height: 1 }, shadowOpacity: 0.05, elevation: 2,
              }}
            >
              <View style={{
                width: 46, height: 52, borderRadius: 10,
                backgroundColor: item.mode === "analyse" ? "#807261" + "15" : "#2980B9" + "15",
                alignItems: "center", justifyContent: "center",
              }}>
                <Ionicons
                  name={item.mode === "analyse" ? "analytics-outline" : "create-outline"}
                  size={22}
                  color={item.mode === "analyse" ? "#807261" : "#2980B9"}
                />
              </View>
              <View style={{ flex: 1 }}>
                <Text numberOfLines={1} style={{ fontSize: 14, fontFamily: "inter-semibold", color: "#1A1A1A" }}>
                  {item.contractType}
                </Text>
                <Text numberOfLines={1} style={{ fontSize: 12, fontFamily: "inter-regular", color: "#9A8E7F", marginTop: 2 }}>
                  {item.fileName}
                </Text>
                <View style={{ flexDirection: "row", alignItems: "center", marginTop: 6, gap: 8 }}>
                  <View style={{
                    backgroundColor: item.mode === "analyse" ? "#807261" + "15" : "#2980B9" + "15",
                    borderRadius: 6, paddingHorizontal: 7, paddingVertical: 2,
                  }}>
                    <Text style={{
                      fontSize: 10, fontFamily: "inter-semibold",
                      color: item.mode === "analyse" ? "#807261" : "#2980B9",
                    }}>
                      {item.mode === "analyse" ? "Analyse" : "Brouillon"}
                    </Text>
                  </View>
                  {item.compliance && (
                    <Text style={{ fontSize: 11, fontFamily: "inter-medium", color: "#555" }}>
                      {item.compliance}
                    </Text>
                  )}
                  <Text style={{ fontSize: 11, fontFamily: "inter-regular", color: "#B0ADA8" }}>
                    {item.date}
                  </Text>
                </View>
              </View>
              <Ionicons name="chevron-forward" size={16} color="#C8C3BD" />
            </TouchableOpacity>
          )}
        />
      )}
    </View>
  );
}

const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));
