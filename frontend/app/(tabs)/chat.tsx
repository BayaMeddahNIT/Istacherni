import {
  Text, View, TextInput, TouchableOpacity, TouchableWithoutFeedback, FlatList,
  KeyboardAvoidingView, Platform, Animated, Alert, Image,
  Modal, Pressable, ActivityIndicator,
} from "react-native";
import { useState, useRef, useEffect, useCallback } from "react";
import { Ionicons } from "@expo/vector-icons";
import { Audio } from "expo-av";
import * as ImagePicker from "expo-image-picker";
import * as DocumentPicker from "expo-document-picker";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { useTheme, useTranslation } from "@/context/UserContext";

// ─── Types ────────────────────────────────────────────────────────────────────

type MessageType = "text" | "voice" | "image" | "file";

interface Message {
  id: string;
  type: MessageType;
  sender: "user" | "bot";
  timestamp: Date;
  text?: string;
  audioUri?: string;
  audioDuration?: number;
  imageUri?: string;
  fileName?: string;
  fileSize?: string;
  isLoading?: boolean;
}

interface Conversation {
  id: string;
  title: string;
  preview: string;
  date: string;
  messages: Message[];
}

// ─── Backend Configuration ───────────────────────────────────────────────────

/**
 * Replace with your machine's LAN IP when testing on a real device.
 * For Android emulator use: http://10.0.2.2:8000
 * For iOS simulator or web use: http://localhost:8000
 * For a real phone on the same WiFi: http://<YOUR_LAN_IP>:8000
 */
//const BACKEND_URL = "http://192.168.1.9:8000"; // your machine's LAN IP
const BACKEND_URL = "https://earthly-backer-obsession.ngrok-free.dev"; // your machine's LAN IP
const HYBRID_ENDPOINT = `${BACKEND_URL}/chat/hybrid`;

// ─── OLD AI Configuration (OpenAI — kept as comment) ─────────────────────────
//
// const AI_CONFIG = {
//   apiKey: "",
//   endpoint: "https://api.openai.com/v1/chat/completions",
//   model: "gpt-3.5-turbo",
// };
//
// const SYSTEM_PROMPT = `Tu es Istacherni, un assistant juridique expert...`;


// ─── Language Detection ───────────────────────────────────────────────────────

function detectLanguage(text: string): "ar" | "fr" | "en" {
  const arabic = /[\u0600-\u06FF\u0750-\u077F]/;
  if (arabic.test(text)) return "ar";
  const frWords = ["je", "vous", "mon", "ma", "le", "la", "les", "de", "du", "est",
    "sont", "avec", "pour", "dans", "que", "qui", "contrat", "droit", "loi", "travail",
    "salaire", "licencié", "tribunal", "justice", "avocat", "procès", "plainte"];
  const lower = text.toLowerCase();
  const frScore = frWords.filter(w => lower.includes(w)).length;
  return frScore >= 1 ? "fr" : "en";
}

// ─── Intelligent Local Legal Responses ───────────────────────────────────────

const LEGAL_KB = {
  fr: [
    {
      keys: ["contrat travail", "contrat de travail", "emploi", "employé", "employeur", "licenci"],
      response: `📋 **Droit du Travail Algérien**

Selon la **Loi n° 90-11** relative aux relations de travail :

• **Art. 10** : Tout contrat de travail doit préciser la durée, la rémunération et la fonction
• **Art. 73** : Le licenciement abusif ouvre droit à des indemnités
• **Art. 87 bis** : Le SNMG (salaire minimum garanti) est obligatoire

⚖️ Pour un licenciement : vous avez 3 mois pour saisir l'inspection du travail.

*Recommandation : Conservez toujours une copie de votre contrat signé.*`
    },
    {
      keys: ["loyer", "bail", "locataire", "propriétaire", "location", "maison", "appartement"],
      response: `🏠 **Droit des Baux en Algérie**

Selon le **Code Civil** (Ordonnance n° 75-58) :

• **Art. 467** : Le bail doit être écrit pour toute durée supérieure à 3 ans
• **Art. 479** : Le propriétaire doit garantir la jouissance paisible du bien
• **Art. 483** : Le locataire doit payer le loyer aux dates convenues

📋 En cas de litige :
1. Tentative de médiation amiable
2. Saisine du tribunal civil (TPI)
3. Délai de recours : 10 ans pour les contrats

*Un acte authentifié chez le notaire est fortement recommandé.*`
    },
    {
      keys: ["divorce", "mariage", "famille", "enfants", "garde", "pension"],
      response: `👨‍👩‍👧 **Code de la Famille Algérien**

Selon la **Loi n° 84-11** portant Code de la Famille :

• **Art. 48** : Le divorce peut être prononcé par le mari ou demandé par la femme (Khul')
• **Art. 62** : La garde des enfants est accordée à la mère jusqu'à 10 ans pour les fils, 16 ans pour les filles
• **Art. 72** : La pension alimentaire est fixée par le juge selon les revenus du père

⚠️ La procédure se déroule devant le **Tribunal de famille** (section du TPI).

*Consultez impérativement un avocat spécialisé en droit de la famille.*`
    },
    {
      keys: ["commerce", "entreprise", "société", "faillite", "registre"],
      response: `🏢 **Droit Commercial Algérien**

Selon le **Code de Commerce** (Ordonnance n° 75-59) :

• **Art. 1** : Tout commerçant doit s'inscrire au Registre du Commerce
• **Art. 215** : La SARL nécessite un capital minimum de 100 000 DA
• **Art. 330** : La faillite est prononcée par le tribunal commercial

📋 Création d'entreprise :
1. Inscription au CNRC (Centre National du RC)
2. Dépôt des statuts chez le notaire
3. Publication au BOAL

*Pour toute création de société, un notaire est obligatoire.*`
    },
    {
      keys: ["pénal", "crime", "délit", "plainte", "tribunal", "arrestation", "prison", "amende"],
      response: `⚖️ **Code Pénal Algérien**

Selon l'**Ordonnance n° 66-156** portant Code Pénal :

• **Art. 2** : La loi pénale s'applique aux infractions commises sur le territoire algérien
• **Art. 42** : Emprisonnement de 10 jours à 10 ans pour les délits
• **Art. 53** : Possibilité de sursis pour les peines ≤ 5 ans

📋 Dépôt de plainte :
1. **Commissariat/Gendarmerie** : PV immédiat
2. **Procureur de la République** : Lettre recommandée
3. **Juge d'instruction** : Plainte avec constitution de partie civile

*Délai de prescription : 3 ans pour les délits, 10 ans pour les crimes.*`
    },
    {
      keys: ["propriété", "terrain", "acte", "notaire", "immobilier", "bien"],
      response: `🏗️ **Droit Immobilier en Algérie**

Selon le **Code Civil** et la **Loi n° 90-25** sur l'orientation foncière :

• Tout transfert de propriété **doit** être établi par acte notarié
• L'acte doit être **publié** à la Conservation Foncière
• Le **certificat de propriété** est délivré par la Conservation Foncière

📋 Documents nécessaires :
- Acte de propriété du vendeur
- Relevé cadastral
- Attestation de non-imposition
- PV de délimitation

*⚠️ Méfiez-vous des ventes sous seing privé — elles ne sont pas opposables aux tiers.*`
    },
    {
      keys: ["héritage", "succession", "testament", "décès", "héritier"],
      response: `📜 **Droit des Successions en Algérie**

Selon le **Code de la Famille** (Loi n° 84-11) et le **droit islamique** :

• **Art. 126-188** : Règles de partage successoral basées sur la Charia
• Le **conjoint survivant** reçoit 1/4 (sans enfants) ou 1/8 (avec enfants)
• Les **filles** reçoivent la moitié de la part des fils

📋 Procédure :
1. Acte de décès + livret de famille
2. Liste des héritiers établie par le notaire
3. Déclaration fiscale de succession (6 mois)
4. Partage notarié

*Délai de déclaration fiscale : 6 mois après le décès.*`
    },
  ],
  ar: [
    {
      keys: ["عقد", "عمل", "موظف", "صاحب عمل", "فصل", "أجر", "راتب"],
      response: `📋 **قانون العمل الجزائري**

وفقاً للـ **القانون رقم 90-11** المتعلق بعلاقات العمل:

• **المادة 10**: يجب أن يتضمن عقد العمل المدة والأجر والمنصب
• **المادة 73**: يُحق للعامل المفصول تعسفياً الحصول على تعويض
• **المادة 87 مكرر**: الأجر الوطني الأدنى المضمون إلزامي

⚖️ في حال الفصل: لديك **3 أشهر** لرفع شكوى لمفتشية العمل.

*نصيحة: احتفظ دائماً بنسخة من عقد العمل الموقع.*`
    },
    {
      keys: ["طلاق", "زواج", "أسرة", "أطفال", "حضانة", "نفقة"],
      response: `👨‍👩‍👧 **قانون الأسرة الجزائري**

وفقاً للـ **القانون رقم 84-11** المتضمن قانون الأسرة:

• **المادة 48**: يحق للزوجة طلب الطلاق بالخلع
• **المادة 62**: الحضانة للأم حتى 10 سنوات للأبناء و16 سنة للبنات
• **المادة 72**: النفقة تُحدد من قبل القاضي بحسب دخل الأب

⚠️ الإجراءات أمام **محكمة الأسرة** (قسم المحكمة الابتدائية)

*يُنصح بشدة باستشارة محامٍ متخصص في قانون الأسرة.*`
    },
    {
      keys: ["جريمة", "شكوى", "شرطة", "اعتقال", "سجن", "غرامة", "قضاء"],
      response: `⚖️ **قانون العقوبات الجزائري**

وفقاً للـ **الأمر رقم 66-156** المتضمن قانون العقوبات:

• **المادة 2**: يُطبق القانون الجزائي على كل جريمة ارتُكبت فوق التراب الجزائري
• **المادة 42**: السجن من 10 أيام إلى 10 سنوات للجنح
• **المادة 53**: إمكانية وقف تنفيذ العقوبة للأحكام ≤ 5 سنوات

📋 **كيفية تقديم شكوى:**
1. **في المركز الأمني/الدرك**: محضر فوري
2. **عند وكيل الجمهورية**: برسالة مضمونة
3. **عند قاضي التحقيق**: شكوى مع ادعاء مدني

*مدة التقادم: 3 سنوات للجنح، 10 سنوات للجنايات.*`
    },
    {
      keys: ["عقار", "ملكية", "أرض", "موثق", "مسكن", "بيع"],
      response: `🏗️ **قانون العقار في الجزائر**

وفقاً للـ **القانون المدني** والـ **قانون رقم 90-25** المتعلق بالتوجيه العقاري:

• أي نقل ملكية **يجب** أن يتم بموجب عقد موثق
• يجب **نشر** العقد في المحافظة العقارية
• **شهادة الملكية** تُسلم من المحافظة العقارية

📋 الوثائق المطلوبة:
- عقد ملكية البائع
- مخطط مساحي
- شهادة عدم الخضوع للضريبة

*⚠️ احذر من عقود البيع العرفية — لا يمكن الاحتجاج بها في مواجهة الغير.*`
    },
  ],
  en: [
    {
      keys: ["contract", "employment", "fired", "worker", "salary", "dismissal"],
      response: `📋 **Algerian Labor Law**

Under **Law No. 90-11** on labor relations:

• **Art. 10**: Employment contracts must specify duration, salary, and position
• **Art. 73**: Unfair dismissal entitles the worker to compensation
• **Art. 87 bis**: The national minimum wage (SNMG) is mandatory

⚖️ If dismissed: you have **3 months** to file a complaint with the Labor Inspectorate.

*Key advice: Always keep a signed copy of your employment contract.*`
    },
    {
      keys: ["criminal", "crime", "complaint", "police", "arrest", "penalty", "court"],
      response: `⚖️ **Algerian Penal Code**

Under **Ordinance No. 66-156** (Penal Code):

• **Art. 2**: Algerian criminal law applies to all offenses committed on Algerian territory
• **Art. 42**: Imprisonment from 10 days to 10 years for misdemeanors
• **Art. 53**: Suspended sentences possible for terms ≤ 5 years

📋 **Filing a complaint:**
1. **Police/Gendarmerie station**: Immediate official report
2. **Public Prosecutor**: Registered letter
3. **Investigation Judge**: Criminal complaint with civil claim

*Statute of limitations: 3 years for misdemeanors, 10 years for crimes.*`
    },
    {
      keys: ["property", "real estate", "land", "notary", "house", "apartment", "buy", "sell"],
      response: `🏗️ **Algerian Real Estate Law**

Under the **Civil Code** and **Law No. 90-25**:

• All property transfers **must** be formalized through a notarial deed
• The deed must be **registered** at the Conservation Office (Conservation Foncière)
• A **title certificate** is required for all transactions

📋 Required documents:
- Seller's title deed
- Cadastral survey
- Non-tax certificate
- Boundary demarcation report

*⚠️ Private sale agreements are not enforceable against third parties.*`
    },
    {
      keys: ["business", "company", "commercial", "register", "bankruptcy", "trade"],
      response: `🏢 **Algerian Commercial Law**

Under the **Commercial Code** (Ordinance No. 75-59):

• **Art. 1**: All merchants must register at the Commercial Registry (CNRC)
• **Art. 215**: SARL (LLC) requires a minimum capital of 100,000 DZD
• **Art. 330**: Bankruptcy is declared by the commercial court

📋 Business registration steps:
1. Register at CNRC
2. Notarize company statutes
3. Publish in the BOAL (Official Gazette)

*A notary is mandatory for company formation.*`
    },
  ],
};

function generateLocalResponse(message: string): string {
  const lang = detectLanguage(message);
  const lower = message.toLowerCase();
  const kb = LEGAL_KB[lang] || LEGAL_KB.fr;

  for (const entry of kb) {
    if (entry.keys.some(k => lower.includes(k))) {
      return entry.response;
    }
  }

  const fallbacks = {
    fr: `Bonjour ! Je suis **Istacherni**, votre assistant juridique spécialisé dans le droit algérien. 🇩🇿

Je peux vous aider sur :
⚖️ Droit du travail et contrats d'emploi
🏠 Droit immobilier et baux
👨‍👩‍👧 Droit de la famille et successions
🏢 Droit commercial et entreprises
📋 Code pénal et procédures judiciaires

Posez-moi votre question juridique et je vous répondrai avec les textes de loi applicables.

*Pour les cas complexes, je vous recommande de consulter un avocat agréé au barreau algérien.*`,
    ar: `مرحباً! أنا **إيستاشيرني**، مساعدك القانوني المتخصص في القانون الجزائري. 🇩🇿

يمكنني مساعدتك في:
⚖️ قانون العمل والعقود
🏠 القانون العقاري والإيجار
👨‍👩‍👧 قانون الأسرة والميراث
🏢 القانون التجاري والشركات
📋 قانون العقوبات والإجراءات القضائية

اطرح سؤالك القانوني وسأجيبك بالنصوص القانونية المعمول بها.

*للحالات المعقدة، أنصحك باستشارة محامٍ مسجل في نقابة المحامين الجزائريين.*`,
    en: `Hello! I'm **Istacherni**, your legal assistant specialized in Algerian law. 🇩🇿

I can help you with:
⚖️ Labor law and employment contracts
🏠 Real estate law and leases
👨‍👩‍👧 Family law and inheritance
🏢 Commercial law and business formation
📋 Criminal code and judicial procedures

Ask me your legal question and I'll respond with the applicable laws.

*For complex cases, I recommend consulting a licensed attorney registered with the Algerian Bar.*`,
  };

  return fallbacks[lang];
}

// ─── Backend API Call (Qwen2.5:7b + BM25 + BGE-M3) ─────────────────────────

async function getAIResponseFromBackend(question: string): Promise<string> {
  /**
   * Sends the user question to the FastAPI /chat/hybrid endpoint.
   * Returns Qwen2.5:7b answer retrieved via BM25 + BAAI/BGE-M3 hybrid RAG.
   *
   * Falls back to local keyword-matching if the backend is unreachable.
   */
  try {
    const res = await fetch(HYBRID_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, top_k: 5 }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      console.warn("[Backend] Non-OK response:", err);
      // fallback to local
      return generateLocalResponse(question);
    }
    const data = await res.json();
    return data.answer as string;
  } catch (e) {
    console.warn("[Backend] Unreachable, using local fallback:", e);
    // Fallback: return local keyword-based response when backend is down
    return generateLocalResponse(question);
  }
}

// ─── OLD getAIResponse (OpenAI + local fallback) — kept as comment ────────────
//
// async function getAIResponse(messages: { role: string; content: string }[]): Promise<string> {
//   if (AI_CONFIG.apiKey) {
//     try {
//       const res = await fetch(AI_CONFIG.endpoint, {
//         method: "POST",
//         headers: {
//           "Content-Type": "application/json",
//           Authorization: `Bearer ${AI_CONFIG.apiKey}`,
//         },
//         body: JSON.stringify({
//           model: AI_CONFIG.model,
//           messages: [{ role: "system", content: SYSTEM_PROMPT }, ...messages],
//           max_tokens: 600,
//           temperature: 0.7,
//         }),
//       });
//       const data = await res.json();
//       return data.choices?.[0]?.message?.content ?? generateLocalResponse(messages[messages.length - 1].content);
//     } catch {
//       return generateLocalResponse(messages[messages.length - 1].content);
//     }
//   }
//   // Simulate realistic network delay
//   await new Promise(r => setTimeout(r, 1200 + Math.random() * 800));
//   return generateLocalResponse(messages[messages.length - 1].content);
// }


// ─── Helpers ─────────────────────────────────────────────────────────────────

function formatDuration(secs: number) {
  const m = Math.floor(secs / 60);
  const s = secs % 60;
  return `${m}:${s.toString().padStart(2, "0")}`;
}

function makeId() {
  return Date.now().toString() + Math.random().toString(36).slice(2);
}

// ─── Typing Indicator ─────────────────────────────────────────────────────────

function TypingIndicator({ theme }: { theme: any }) {
  const dots = [useRef(new Animated.Value(0.3)).current, useRef(new Animated.Value(0.3)).current, useRef(new Animated.Value(0.3)).current];
  useEffect(() => {
    dots.forEach((dot, i) => {
      Animated.loop(
        Animated.sequence([
          Animated.delay(i * 200),
          Animated.timing(dot, { toValue: 1, duration: 400, useNativeDriver: true }),
          Animated.timing(dot, { toValue: 0.3, duration: 400, useNativeDriver: true }),
        ])
      ).start();
    });
  }, []);

  return (
    <View style={{ alignSelf: "flex-start", marginHorizontal: 16, marginVertical: 6 }}>
      <View style={{ backgroundColor: theme.primary, borderRadius: 20, borderBottomLeftRadius: 6, paddingHorizontal: 16, paddingVertical: 12, flexDirection: "row", gap: 4, alignItems: "center" }}>
        {dots.map((dot, i) => (
          <Animated.View key={i} style={{ width: 7, height: 7, borderRadius: 4, backgroundColor: "rgba(255,255,255,0.9)", opacity: dot }} />
        ))}
      </View>
    </View>
  );
}

// ─── Message Bubble ───────────────────────────────────────────────────────────

function MessageBubble({ msg, theme }: { msg: Message; theme: any }) {
  const isUser = msg.sender === "user";
  const bgColor = isUser ? theme.card : theme.primary;
  const textColor = isUser ? theme.text : "#FFFFFF";

  function formatText(text: string) {
    // Bold (**text**) and line breaks
    const parts = text.split(/(\*\*[^*]+\*\*)/g);
    return parts.map((part, i) => {
      if (part.startsWith("**") && part.endsWith("**")) {
        return <Text key={i} style={{ fontFamily: "inter-semibold" }}>{part.slice(2, -2)}</Text>;
      }
      return <Text key={i}>{part}</Text>;
    });
  }

  return (
    <View style={{ alignSelf: isUser ? "flex-end" : "flex-start", maxWidth: "82%", marginVertical: 5, marginHorizontal: 14 }}>
      <View style={{
        backgroundColor: bgColor, borderRadius: 18,
        borderBottomRightRadius: isUser ? 4 : 18,
        borderBottomLeftRadius: isUser ? 18 : 4,
        paddingHorizontal: 14, paddingVertical: 10,
        shadowColor: theme.shadow, shadowOffset: { width: 0, height: 1 }, shadowOpacity: 0.07, elevation: 2,
      }}>
        {msg.type === "text" && (
          <Text style={{ fontSize: 15, lineHeight: 22, color: textColor, fontFamily: "inter-regular" }}>
            {formatText(msg.text!)}
          </Text>
        )}
        {msg.type === "voice" && (
          <View style={{ flexDirection: "row", alignItems: "center", gap: 10, minWidth: 120 }}>
            <Ionicons name="mic" size={18} color={textColor} />
            <View style={{ flex: 1, height: 3, backgroundColor: isUser ? theme.primary + "40" : "rgba(255,255,255,0.4)", borderRadius: 2 }}>
              <View style={{ width: "60%", height: "100%", backgroundColor: isUser ? theme.primary : "rgba(255,255,255,0.8)", borderRadius: 2 }} />
            </View>
            <Text style={{ fontSize: 12, color: textColor, opacity: 0.8 }}>
              {formatDuration(msg.audioDuration ?? 0)}
            </Text>
          </View>
        )}
        {msg.type === "image" && msg.imageUri && (
          <View>
            <Image source={{ uri: msg.imageUri }} style={{ width: 200, height: 150, borderRadius: 12 }} resizeMode="cover" />
          </View>
        )}
        {msg.type === "file" && (
          <View style={{ flexDirection: "row", alignItems: "center", gap: 10 }}>
            <View style={{ width: 36, height: 36, borderRadius: 10, backgroundColor: isUser ? theme.primary + "20" : "rgba(255,255,255,0.2)", alignItems: "center", justifyContent: "center" }}>
              <Ionicons name="document-text" size={20} color={textColor} />
            </View>
            <View style={{ flex: 1 }}>
              <Text style={{ fontSize: 13, fontFamily: "inter-semibold", color: textColor }} numberOfLines={1}>{msg.fileName}</Text>
              {msg.fileSize && <Text style={{ fontSize: 11, color: textColor, opacity: 0.7 }}>{msg.fileSize}</Text>}
            </View>
          </View>
        )}
      </View>
      <Text style={{ fontSize: 10, color: theme.textMuted, marginTop: 3, textAlign: isUser ? "right" : "left", paddingHorizontal: 4 }}>
        {new Date(msg.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
      </Text>
    </View>
  );
}

// ─── Main Screen ──────────────────────────────────────────────────────────────

export default function Chat() {
  const theme = useTheme();
  const { t, isRTL } = useTranslation();

  // Messages & conversations
  const [messages, setMessages] = useState<Message[]>([]);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [conversationId, setConversationId] = useState(makeId());

  // UI state
  const [inputText, setInputText] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [showAttachSheet, setShowAttachSheet] = useState(false);
  const [showSidebar, setShowSidebar] = useState(false);

  // Pending attachment (staged before sending)
  const [pendingAttachment, setPendingAttachment] = useState<{
    type: "image" | "file";
    uri?: string;
    name?: string;
    size?: string;
  } | null>(null);

  // Voice recording
  const [isRecording, setIsRecording] = useState(false);
  const [recordingDuration, setRecordingDuration] = useState(0);
  const recordingRef = useRef<Audio.Recording | null>(null);
  const recordingTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const pulseAnim = useRef(new Animated.Value(1)).current;

  // Refs
  const flatListRef = useRef<FlatList>(null);
  const sidebarAnim = useRef(new Animated.Value(-320)).current;
  const aiMessages = useRef<{ role: string; content: string }[]>([]);

  // ── Load history ──────────────────────────────────────────────────────────

  useEffect(() => {
    loadConversations();
  }, []);

  const loadConversations = async () => {
    try {
      const raw = await AsyncStorage.getItem("@chat_history");
      if (raw) setConversations(JSON.parse(raw));
    } catch { }
  };

  const saveCurrentConversation = async (msgs: Message[]) => {
    if (msgs.length < 2) return;
    const firstMsg = msgs.find(m => m.sender === "user");
    const conv: Conversation = {
      id: conversationId,
      title: firstMsg?.text?.slice(0, 40) || "Conversation",
      preview: msgs[msgs.length - 1]?.text?.slice(0, 60) || "...",
      date: new Date().toLocaleDateString("fr-DZ"),
      messages: msgs,
    };
    const updated = [conv, ...conversations.filter(c => c.id !== conversationId)].slice(0, 20);
    setConversations(updated);
    await AsyncStorage.setItem("@chat_history", JSON.stringify(updated));
  };

  // ── Sidebar ───────────────────────────────────────────────────────────────

  const openSidebar = () => {
    setShowSidebar(true);
    Animated.spring(sidebarAnim, { toValue: 0, useNativeDriver: true, damping: 20, stiffness: 150 }).start();
  };

  const closeSidebar = () => {
    Animated.timing(sidebarAnim, { toValue: -320, duration: 220, useNativeDriver: true }).start(() => setShowSidebar(false));
  };

  const loadConversation = (conv: Conversation) => {
    setMessages(conv.messages);
    setConversationId(conv.id);
    aiMessages.current = conv.messages
      .filter(m => m.type === "text" && m.text)
      .map(m => ({ role: m.sender === "user" ? "user" : "assistant", content: m.text! }));
    closeSidebar();
  };

  const deleteConversation = (conv: Conversation) => {
    Alert.alert(
      "Supprimer cette conversation ?",
      `"${conv.title}" sera définitivement supprimée.`,
      [
        { text: "Annuler", style: "cancel" },
        {
          text: "Supprimer",
          style: "destructive",
          onPress: async () => {
            const updated = conversations.filter(c => c.id !== conv.id);
            setConversations(updated);
            await AsyncStorage.setItem("@chat_history", JSON.stringify(updated));
            // If deleting the currently displayed conversation, reset
            if (conv.id === conversationId) {
              setMessages([]);
              aiMessages.current = [];
              setConversationId(makeId());
            }
          },
        },
      ]
    );
  };

  const deleteAllConversations = () => {
    if (conversations.length === 0) return;
    Alert.alert(
      "Supprimer tout l'historique ?",
      "Toutes vos conversations seront définitivement effacées. Cette action est irréversible.",
      [
        { text: "Annuler", style: "cancel" },
        {
          text: "Tout supprimer",
          style: "destructive",
          onPress: async () => {
            setConversations([]);
            await AsyncStorage.removeItem("@chat_history");
            // Reset active chat to empty
            setMessages([]);
            aiMessages.current = [];
            setConversationId(makeId());
          },
        },
      ]
    );
  };

  // ── New Chat ──────────────────────────────────────────────────────────────

  const startNewChat = () => {
    if (messages.length > 0) saveCurrentConversation(messages);
    setMessages([]);
    aiMessages.current = [];
    setConversationId(makeId());
  };

  // ── Send Message (with optional pending attachment) ─────────────────────

  const sendMessage = useCallback(async (text: string) => {
    const hasText = text.trim().length > 0;
    const hasAttachment = pendingAttachment !== null;
    if (!hasText && !hasAttachment) return;

    const newMsgs: Message[] = [];

    // Add attachment message first if one is staged
    if (hasAttachment) {
      if (pendingAttachment!.type === "image") {
        newMsgs.push({ id: makeId(), type: "image", sender: "user", timestamp: new Date(), imageUri: pendingAttachment!.uri });
      } else {
        newMsgs.push({ id: makeId(), type: "file", sender: "user", timestamp: new Date(), fileName: pendingAttachment!.name, fileSize: pendingAttachment!.size });
      }
      setPendingAttachment(null);
    }

    // Add text message if present
    if (hasText) {
      newMsgs.push({ id: makeId(), type: "text", sender: "user", timestamp: new Date(), text: text.trim() });
      aiMessages.current.push({ role: "user", content: text.trim() });
    }

    const updated = [...messages, ...newMsgs];
    setMessages(updated);
    setInputText("");
    setTimeout(() => flatListRef.current?.scrollToEnd({ animated: true }), 100);

    // Build AI context
    const aiContent = hasAttachment && !hasText
      ? (pendingAttachment?.type === "image"
        ? "[Image/document partagé pour analyse juridique]"
        : `[Document partagé: ${pendingAttachment?.name}. Analysez ce document juridique.]`)
      : hasAttachment && hasText
      ? `${text.trim()}\n\n[Fichier joint: ${pendingAttachment?.type === "image" ? "Image" : pendingAttachment?.name}]`
      : text.trim();

    setIsTyping(true);
    try {
      // ── NEW: call /chat/hybrid backend (Qwen2.5:7b + BM25 + BGE-M3) ──────
      const questionText = hasText ? text.trim() : aiContent;
      const reply = await getAIResponseFromBackend(questionText);

      // ── OLD: OpenAI / local fallback call — kept as comment ───────────────
      // const aiCtx = hasText
      //   ? [...aiMessages.current]
      //   : [...aiMessages.current, { role: "user", content: aiContent }];
      // const reply = await getAIResponse(aiCtx);

      const botMsg: Message = { id: makeId(), type: "text", sender: "bot", timestamp: new Date(), text: reply };
      const finalMsgs = [...updated, botMsg];
      setMessages(finalMsgs);
      aiMessages.current.push({ role: "assistant", content: reply });
      setTimeout(() => flatListRef.current?.scrollToEnd({ animated: true }), 100);
      await saveCurrentConversation(finalMsgs);
    } catch {
      const errMsg: Message = { id: makeId(), type: "text", sender: "bot", timestamp: new Date(), text: "❌ Une erreur est survenue. Veuillez réessayer." };
      setMessages(prev => [...prev, errMsg]);
    } finally {
      setIsTyping(false);
    }
  }, [messages, pendingAttachment]);

  // ── Voice Recording ───────────────────────────────────────────────────────

  const startRecording = async () => {
    try {
      const { granted } = await Audio.requestPermissionsAsync();
      if (!granted) { Alert.alert("Permission requise", "Veuillez autoriser l'accès au microphone dans les paramètres."); return; }
      await Audio.setAudioModeAsync({ allowsRecordingIOS: true, playsInSilentModeIOS: true });
      const { recording } = await Audio.Recording.createAsync(Audio.RecordingOptionsPresets.HIGH_QUALITY);
      recordingRef.current = recording;
      setIsRecording(true);
      setRecordingDuration(0);
      recordingTimerRef.current = setInterval(() => setRecordingDuration(d => d + 1), 1000);
      Animated.loop(
        Animated.sequence([
          Animated.timing(pulseAnim, { toValue: 1.3, duration: 600, useNativeDriver: true }),
          Animated.timing(pulseAnim, { toValue: 1, duration: 600, useNativeDriver: true }),
        ])
      ).start();
    } catch {
      Alert.alert("Erreur", "Impossible de démarrer l'enregistrement.");
    }
  };

  const stopRecording = async (send = true) => {
    if (!recordingRef.current) return;
    if (recordingTimerRef.current) clearInterval(recordingTimerRef.current);
    pulseAnim.stopAnimation();
    pulseAnim.setValue(1);
    try {
      await recordingRef.current.stopAndUnloadAsync();
      const uri = recordingRef.current.getURI();
      const duration = recordingDuration;
      recordingRef.current = null;
      setIsRecording(false);
      setRecordingDuration(0);
      if (send && uri && duration > 0) {
        const voiceMsg: Message = { id: makeId(), type: "voice", sender: "user", timestamp: new Date(), audioUri: uri, audioDuration: duration };
        const updated = [...messages, voiceMsg];
        setMessages(updated);
        setTimeout(() => flatListRef.current?.scrollToEnd({ animated: true }), 100);
        // AI response to voice message
        setIsTyping(true);
        const reply = await getAIResponseFromBackend("[Message vocal envoyé]");
        const botMsg: Message = { id: makeId(), type: "text", sender: "bot", timestamp: new Date(), text: reply };
        const finalMsgs = [...updated, botMsg];
        setMessages(finalMsgs);
        setIsTyping(false);
        await saveCurrentConversation(finalMsgs);
      }
    } catch { setIsRecording(false); setRecordingDuration(0); }
  };

  // ── Attachments — Stage first, send later ────────────────────────────────

  /**
   * Close the bottom sheet FIRST, then wait for the modal animation to finish
   * before calling any native picker. Calling both at the same time causes a
   * silent race condition where the picker never opens.
   */
  const closeSheetThen = (fn: () => void, delay = 350) => {
    setShowAttachSheet(false);
    setTimeout(fn, delay);
  };

  const takePicture = () => closeSheetThen(async () => {
    try {
      const perm = await ImagePicker.requestCameraPermissionsAsync();
      if (perm.status !== "granted") {
        Alert.alert(
          "Permission refusée",
          "L'accès à l'appareil photo est requis. Activez-le dans les paramètres de l'application.",
          [{ text: "OK" }]
        );
        return;
      }
      const result = await ImagePicker.launchCameraAsync({
        quality: 0.85,
        allowsEditing: true,
        aspect: [4, 3],
      });
      if (!result.canceled && result.assets?.[0]) {
        setPendingAttachment({ type: "image", uri: result.assets[0].uri });
      }
    } catch (e) {
      console.log("[Camera error]", e);
      Alert.alert("Erreur", "Impossible d'accéder à l'appareil photo.");
    }
  });

  const pickImage = () => closeSheetThen(async () => {
    try {
      const perm = await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (perm.status !== "granted") {
        Alert.alert(
          "Permission refusée",
          "L'accès à la galerie est requis. Activez-le dans les paramètres de l'application.",
          [{ text: "OK" }]
        );
        return;
      }
      const result = await ImagePicker.launchImageLibraryAsync({
        quality: 0.85,
        allowsEditing: true,
        // Use string array (new API) – avoids the deprecated MediaTypeOptions warning
        mediaTypes: ["images"] as any,
      });
      if (!result.canceled && result.assets?.[0]) {
        setPendingAttachment({ type: "image", uri: result.assets[0].uri });
      }
    } catch (e) {
      console.log("[Gallery error]", e);
      Alert.alert("Erreur", "Impossible d'accéder à la galerie.");
    }
  });

  const pickDocument = () => closeSheetThen(async () => {
    try {
      const result = await DocumentPicker.getDocumentAsync({
        type: ["application/pdf", "*/*"],
        copyToCacheDirectory: true,
        multiple: false,
      });
      if (!result.canceled && result.assets?.[0]) {
        const asset = result.assets[0];
        const sizeKB = asset.size ? `${Math.round(asset.size / 1024)} KB` : "";
        setPendingAttachment({ type: "file", uri: asset.uri, name: asset.name, size: sizeKB });
      }
    } catch (e) {
      console.log("[Document error]", e);
      Alert.alert("Erreur", "Impossible de sélectionner le fichier.");
    }
  });

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <View style={{ flex: 1, backgroundColor: theme.background }}>

      {/* ── Sidebar Overlay ── */}
      {showSidebar && (
        <Pressable onPress={closeSidebar} style={{ position: "absolute", top: 0, left: 0, right: 0, bottom: 0, backgroundColor: "rgba(0,0,0,0.4)", zIndex: 10 }}>
          <Animated.View
            style={{
              position: "absolute", top: 0, left: 0, bottom: 0, width: 300,
              backgroundColor: theme.card, zIndex: 11,
              transform: [{ translateX: sidebarAnim }],
              shadowColor: "#000", shadowOffset: { width: 4, height: 0 }, shadowOpacity: 0.2, elevation: 20,
            }}
          >
            <Pressable onPress={() => {}} style={{ flex: 1 }}>
              {/* Sidebar Header */}
              <View style={{ paddingTop: Platform.OS === "ios" ? 56 : 48, paddingHorizontal: 20, paddingBottom: 16, backgroundColor: theme.headerBg }}>
                <View style={{ flexDirection: "row", alignItems: "center", justifyContent: "space-between", marginBottom: 4 }}>
                  <Text style={{ fontSize: 18, fontFamily: "inter-semibold", color: theme.text }}>Historique</Text>
                  {conversations.length > 0 && (
                    <TouchableOpacity
                      onPress={deleteAllConversations}
                      hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
                      style={{
                        flexDirection: "row", alignItems: "center", gap: 4,
                        backgroundColor: theme.danger + "14", borderRadius: 8,
                        paddingHorizontal: 8, paddingVertical: 5,
                      }}
                    >
                      <Ionicons name="trash-outline" size={13} color={theme.danger} />
                      <Text style={{ fontSize: 11, fontFamily: "inter-semibold", color: theme.danger }}>Tout effacer</Text>
                    </TouchableOpacity>
                  )}
                </View>
                <Text style={{ fontSize: 12, color: theme.textSecondary }}>
                  {conversations.length === 0 ? "Aucune conversation" : `${conversations.length} conversation${conversations.length > 1 ? "s" : ""}`}
                </Text>
              </View>

              {/* Conversation List */}
              <FlatList
                data={conversations}
                keyExtractor={c => c.id}
                ListEmptyComponent={
                  <View style={{ alignItems: "center", padding: 40 }}>
                    <Ionicons name="chatbubbles-outline" size={40} color={theme.textMuted} />
                    <Text style={{ fontSize: 14, color: theme.textMuted, marginTop: 12, textAlign: "center" }}>
                      Aucune conversation pour l'instant
                    </Text>
                  </View>
                }
                renderItem={({ item }) => (
                  <View style={{
                    flexDirection: "row", alignItems: "center",
                    borderBottomWidth: 0.5, borderBottomColor: theme.divider,
                    backgroundColor: item.id === conversationId ? theme.primaryLight : "transparent",
                  }}>
                    {/* Conversation info — tap to open */}
                    <TouchableOpacity
                      onPress={() => loadConversation(item)}
                      style={{ flex: 1, paddingHorizontal: 20, paddingVertical: 14 }}
                    >
                      <Text style={{ fontSize: 14, fontFamily: "inter-semibold", color: theme.text }} numberOfLines={1}>
                        {item.title}
                      </Text>
                      <Text style={{ fontSize: 12, color: theme.textSecondary, marginTop: 3 }} numberOfLines={1}>
                        {item.preview}
                      </Text>
                      <Text style={{ fontSize: 10, color: theme.textMuted, marginTop: 4 }}>{item.date}</Text>
                    </TouchableOpacity>

                    {/* Per-item delete button */}
                    <TouchableOpacity
                      onPress={() => deleteConversation(item)}
                      hitSlop={{ top: 10, bottom: 10, left: 6, right: 6 }}
                      style={{
                        paddingHorizontal: 14, paddingVertical: 12,
                        alignItems: "center", justifyContent: "center",
                      }}
                    >
                      <View style={{
                        width: 30, height: 30, borderRadius: 8,
                        backgroundColor: theme.danger + "12",
                        alignItems: "center", justifyContent: "center",
                      }}>
                        <Ionicons name="trash-outline" size={15} color={theme.danger} />
                      </View>
                    </TouchableOpacity>
                  </View>
                )}
              />

              {/* New Chat button */}
              <TouchableOpacity
                onPress={() => { closeSidebar(); startNewChat(); }}
                style={{ margin: 16, backgroundColor: theme.primary, borderRadius: 14, paddingVertical: 14, flexDirection: "row", alignItems: "center", justifyContent: "center", gap: 8 }}
              >
                <Ionicons name="add-circle-outline" size={18} color="#fff" />
                <Text style={{ fontSize: 14, fontFamily: "inter-semibold", color: "#fff" }}>Nouvelle conversation</Text>
              </TouchableOpacity>
            </Pressable>
          </Animated.View>
        </Pressable>
      )}

      <KeyboardAvoidingView
        behavior={Platform.OS === "ios" ? "padding" : "height"}
        style={{ flex: 1 }}
        keyboardVerticalOffset={Platform.OS === "ios" ? 0 : 0}
      >
        {/* ── Header ── */}
        <View style={{
          paddingTop: Platform.OS === "ios" ? 56 : 48, paddingHorizontal: 16, paddingBottom: 14,
          backgroundColor: theme.headerBg,
          borderBottomLeftRadius: 20, borderBottomRightRadius: 20,
          flexDirection: "row", alignItems: "center", justifyContent: "space-between",
        }}>
          <TouchableOpacity onPress={openSidebar} style={{ width: 40, height: 40, borderRadius: 12, backgroundColor: "rgba(255,255,255,0.3)", alignItems: "center", justifyContent: "center" }}>
            <Ionicons name="menu" size={22} color={theme.text} />
          </TouchableOpacity>

          <View style={{ alignItems: "center" }}>
            <View style={{ flexDirection: "row", alignItems: "center", gap: 6 }}>
              <View style={{ width: 8, height: 8, borderRadius: 4, backgroundColor: theme.success }} />
              <Text style={{ fontSize: 16, fontFamily: "inter-semibold", color: theme.text }}>Istacherni IA</Text>
            </View>
            <Text style={{ fontSize: 11, color: theme.primary, marginTop: 1 }}>Assistant Juridique Algérien</Text>
          </View>

          <TouchableOpacity
            onPress={startNewChat}
            style={{ width: 40, height: 40, borderRadius: 12, backgroundColor: "rgba(255,255,255,0.3)", alignItems: "center", justifyContent: "center" }}
          >
            <Ionicons name="add" size={22} color={theme.text} />
          </TouchableOpacity>
        </View>

        {/* ── Empty State ── */}
        {messages.length === 0 && !isTyping && (
          <View style={{ flex: 1, alignItems: "center", justifyContent: "center", padding: 32 }}>
            <View style={{ width: 72, height: 72, borderRadius: 24, backgroundColor: theme.primaryLight, alignItems: "center", justifyContent: "center", marginBottom: 20 }}>
              <Ionicons name="chatbubbles-outline" size={36} color={theme.primary} />
            </View>
            <Text style={{ fontSize: 20, fontFamily: "inter-semibold", color: theme.text, textAlign: "center", marginBottom: 10 }}>
              {isRTL ? "كيف يمكنني مساعدتك؟" : "Comment puis-je vous aider ?"}
            </Text>
            <Text style={{ fontSize: 13, color: theme.textSecondary, textAlign: "center", lineHeight: 20, marginBottom: 28 }}>
              {isRTL ? "اطرح سؤالك القانوني وسأجيبك بناءً على القانون الجزائري." : "Posez votre question juridique, je réponds selon la législation algérienne."}
            </Text>
            <View style={{ gap: 10, width: "100%" }}>
              {[
                isRTL ? "ما هي حقوقي عند الفصل من العمل؟" : "Quels sont mes droits en cas de licenciement ?",
                isRTL ? "كيف أقدم شكوى أمام المحكمة؟" : "Comment déposer une plainte au tribunal ?",
                isRTL ? "كيف أؤسس شركة في الجزائر؟" : "Comment créer une entreprise en Algérie ?",
              ].map((q, i) => (
                <TouchableOpacity key={i} onPress={() => sendMessage(q)} style={{
                  backgroundColor: theme.card, borderRadius: 14, padding: 14,
                  flexDirection: "row", alignItems: "center", justifyContent: "space-between",
                  borderWidth: 1, borderColor: theme.border,
                  shadowColor: theme.shadow, shadowOffset: { width: 0, height: 1 }, shadowOpacity: 0.05, elevation: 1,
                }}>
                  <Text style={{ flex: 1, fontSize: 13, color: theme.text, fontFamily: "inter-regular" }}>{q}</Text>
                  <Ionicons name="chevron-forward" size={16} color={theme.textMuted} />
                </TouchableOpacity>
              ))}
            </View>
          </View>
        )}

        {/* ── Messages ── */}
        {messages.length > 0 && (
          <FlatList
            ref={flatListRef}
            data={messages}
            keyExtractor={m => m.id}
            renderItem={({ item }) => <MessageBubble msg={item} theme={theme} />}
            contentContainerStyle={{ paddingVertical: 12, paddingBottom: 6 }}
            showsVerticalScrollIndicator={false}
            onContentSizeChange={() => flatListRef.current?.scrollToEnd({ animated: true })}
            keyboardShouldPersistTaps="handled"
            ListFooterComponent={isTyping ? <TypingIndicator theme={theme} /> : null}
          />
        )}
        {messages.length === 0 && isTyping && <TypingIndicator theme={theme} />}

        {/* ── Voice Recording Overlay ── */}
        {isRecording && (
          <View style={{
            position: "absolute", bottom: 0, left: 0, right: 0,
            backgroundColor: theme.card,
            borderTopLeftRadius: 24, borderTopRightRadius: 24,
            padding: 24, alignItems: "center", gap: 14,
            shadowColor: "#000", shadowOffset: { width: 0, height: -4 }, shadowOpacity: 0.12, elevation: 16,
          }}>
            <Animated.View style={{ transform: [{ scale: pulseAnim }] }}>
              <View style={{ width: 64, height: 64, borderRadius: 32, backgroundColor: theme.danger + "20", alignItems: "center", justifyContent: "center" }}>
                <Ionicons name="mic" size={32} color={theme.danger} />
              </View>
            </Animated.View>
            <Text style={{ fontSize: 28, fontFamily: "inter-semibold", color: theme.danger }}>{formatDuration(recordingDuration)}</Text>
            <Text style={{ fontSize: 13, color: theme.textSecondary }}>Enregistrement en cours…</Text>
            <View style={{ flexDirection: "row", gap: 16 }}>
              <TouchableOpacity onPress={() => stopRecording(false)} style={{ flex: 1, backgroundColor: theme.pillBg, borderRadius: 14, paddingVertical: 14, alignItems: "center" }}>
                <Text style={{ fontSize: 14, fontFamily: "inter-semibold", color: theme.textSecondary }}>Annuler</Text>
              </TouchableOpacity>
              <TouchableOpacity onPress={() => stopRecording(true)} style={{ flex: 1, backgroundColor: theme.danger, borderRadius: 14, paddingVertical: 14, flexDirection: "row", alignItems: "center", justifyContent: "center", gap: 8 }}>
                <Ionicons name="send" size={16} color="#fff" />
                <Text style={{ fontSize: 14, fontFamily: "inter-semibold", color: "#fff" }}>Envoyer</Text>
              </TouchableOpacity>
            </View>
          </View>
        )}

        {/* ── Input Bar ── */}
        {!isRecording && (
          <View style={{ backgroundColor: theme.headerBg, paddingBottom: Platform.OS === "ios" ? 28 : 12 }}>

            {/* Attachment Preview Strip */}
            {pendingAttachment && (
              <View style={{
                flexDirection: "row", alignItems: "center",
                marginHorizontal: 12, marginBottom: 8, marginTop: 4,
                backgroundColor: theme.card, borderRadius: 16, padding: 10, gap: 10,
                borderWidth: 1, borderColor: theme.primaryMedium,
                shadowColor: theme.shadow, shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.07, elevation: 3,
              }}>
                {/* Preview thumbnail or file icon */}
                {pendingAttachment.type === "image" && pendingAttachment.uri ? (
                  <Image
                    source={{ uri: pendingAttachment.uri }}
                    style={{ width: 52, height: 52, borderRadius: 10 }}
                    resizeMode="cover"
                  />
                ) : (
                  <View style={{ width: 52, height: 52, borderRadius: 10, backgroundColor: "#C0392B15", alignItems: "center", justifyContent: "center", borderWidth: 1, borderColor: "#C0392B25" }}>
                    <Ionicons name="document-text" size={24} color="#C0392B" />
                  </View>
                )}

                {/* File info */}
                <View style={{ flex: 1 }}>
                  <Text style={{ fontSize: 13, fontFamily: "inter-semibold", color: theme.text }} numberOfLines={1}>
                    {pendingAttachment.type === "image" ? "Image sélectionnée" : pendingAttachment.name}
                  </Text>
                  <Text style={{ fontSize: 11, color: theme.textSecondary, marginTop: 2 }}>
                    {pendingAttachment.type === "image" ? "Prêt à envoyer ✓" : `${pendingAttachment.size} • Prêt à envoyer ✓`}
                  </Text>
                  {pendingAttachment.type === "file" && (
                    <Text style={{ fontSize: 10, color: theme.primary, marginTop: 2, fontFamily: "inter-medium" }}>
                      📋 Analyse juridique en attente
                    </Text>
                  )}
                </View>

                {/* Cancel attachment */}
                <TouchableOpacity
                  onPress={() => setPendingAttachment(null)}
                  hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
                  style={{
                    width: 28, height: 28, borderRadius: 14,
                    backgroundColor: theme.danger + "15",
                    alignItems: "center", justifyContent: "center",
                  }}
                >
                  <Ionicons name="close" size={16} color={theme.danger} />
                </TouchableOpacity>
              </View>
            )}

            {/* Text input row */}
            <View style={{
              flexDirection: "row", alignItems: "center",
              paddingHorizontal: 12, paddingVertical: 6, gap: 8,
            }}>
              <TouchableOpacity onPress={() => setShowAttachSheet(true)} style={{
                width: 40, height: 40, borderRadius: 20,
                borderWidth: 1.5, borderColor: pendingAttachment ? theme.primary : theme.primary,
                backgroundColor: pendingAttachment ? theme.primaryLight : "transparent",
                alignItems: "center", justifyContent: "center",
              }}>
                <Ionicons name={pendingAttachment ? "attach" : "add"} size={22} color={theme.primary} />
              </TouchableOpacity>

              <View style={{
                flex: 1, flexDirection: "row", alignItems: "center",
                backgroundColor: theme.card, borderRadius: 24, paddingHorizontal: 14, minHeight: 44,
                shadowColor: theme.shadow, shadowOffset: { width: 0, height: 1 }, shadowOpacity: 0.05, elevation: 1,
              }}>
                <TextInput
                  placeholder={pendingAttachment ? "Ajouter un message (optionnel)…" : t("chatPlaceholder")}
                  placeholderTextColor={theme.textMuted}
                  value={inputText}
                  onChangeText={setInputText}
                  style={{ flex: 1, fontSize: 15, color: theme.text, paddingVertical: 8, textAlign: isRTL ? "right" : "left" }}
                  multiline
                  maxLength={1000}
                  returnKeyType="default"
                />
                <TouchableOpacity onPress={startRecording} hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}>
                  <Ionicons name="mic-outline" size={22} color={theme.primary} />
                </TouchableOpacity>
              </View>

              <TouchableOpacity
                onPress={() => sendMessage(inputText)}
                disabled={(!inputText.trim() && !pendingAttachment) || isTyping}
                style={{
                  width: 44, height: 44, borderRadius: 22,
                  backgroundColor: (inputText.trim() || pendingAttachment) && !isTyping ? theme.primary : theme.divider,
                  alignItems: "center", justifyContent: "center",
                }}
              >
                {isTyping
                  ? <ActivityIndicator size="small" color="#fff" />
                  : <Ionicons name="send" size={18} color="#FFFFFF" />
                }
              </TouchableOpacity>
            </View>
          </View>
        )}
      </KeyboardAvoidingView>

      {/* ── Attachment Bottom Sheet ── */}
      <Modal
        visible={showAttachSheet}
        transparent
        animationType="slide"
        onRequestClose={() => setShowAttachSheet(false)}
        statusBarTranslucent
      >
        {/*
         * Layout: full-screen View, with a transparent tap-to-close area on top,
         * and the sheet anchored at the bottom. We do NOT use nested Pressables
         * because that pattern is known to swallow child onPress events.
         */}
        <View style={{ flex: 1, justifyContent: "flex-end" }}>
          {/* Transparent backdrop — tapping it closes the sheet */}
          <TouchableWithoutFeedback onPress={() => setShowAttachSheet(false)}>
            <View style={{ position: "absolute", top: 0, left: 0, right: 0, bottom: 0, backgroundColor: "rgba(0,0,0,0.45)" }} />
          </TouchableWithoutFeedback>

          {/* Sheet content — sits on top of the backdrop */}
          <View style={{
            backgroundColor: theme.card,
            borderTopLeftRadius: 28, borderTopRightRadius: 28,
            padding: 24,
            paddingBottom: Platform.OS === "ios" ? 44 : 28,
          }}>
            {/* Handle */}
            <View style={{ width: 40, height: 4, backgroundColor: theme.divider, borderRadius: 2, alignSelf: "center", marginBottom: 20 }} />

            <Text style={{ fontSize: 16, fontFamily: "inter-semibold", color: theme.text, marginBottom: 24, textAlign: "center" }}>
              Joindre un fichier
            </Text>

            {/* Option buttons */}
            <View style={{ flexDirection: "row", justifyContent: "space-around" }}>
              {[
                { icon: "camera", label: "Caméra", action: takePicture, color: "#E67E22", bg: "#E67E2215" },
                { icon: "image", label: "Galerie", action: pickImage, color: "#2980B9", bg: "#2980B915" },
                { icon: "document-text", label: "Fichier PDF", action: pickDocument, color: "#C0392B", bg: "#C0392B15" },
              ].map((opt) => (
                <TouchableOpacity
                  key={opt.label}
                  onPress={opt.action}
                  activeOpacity={0.65}
                  style={{ alignItems: "center", gap: 10, flex: 1 }}
                >
                  <View style={{
                    width: 64, height: 64, borderRadius: 20,
                    backgroundColor: opt.bg,
                    alignItems: "center", justifyContent: "center",
                    borderWidth: 1.5, borderColor: opt.color + "30",
                  }}>
                    <Ionicons name={opt.icon as any} size={28} color={opt.color} />
                  </View>
                  <Text style={{ fontSize: 12, fontFamily: "inter-semibold", color: theme.textSecondary }}>
                    {opt.label}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>

            <TouchableOpacity
              onPress={() => setShowAttachSheet(false)}
              style={{
                marginTop: 24, paddingVertical: 14, borderRadius: 14,
                backgroundColor: theme.pillBg, alignItems: "center",
              }}
            >
              <Text style={{ fontSize: 14, color: theme.textSecondary, fontFamily: "inter-semibold" }}>Annuler</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>
    </View>
  );
}
