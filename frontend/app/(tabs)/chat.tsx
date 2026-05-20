import {
  Text, View, TextInput, TouchableOpacity, TouchableWithoutFeedback, FlatList,
  KeyboardAvoidingView, Platform, Animated, Alert, Image,
  Modal, Pressable, ActivityIndicator, Keyboard,
} from "react-native";
import * as Clipboard from "expo-clipboard";
import { useState, useRef, useEffect, useCallback } from "react";
import { Ionicons } from "@expo/vector-icons";
import { Audio } from "expo-av";
import * as ImagePicker from "expo-image-picker";
import * as DocumentPicker from "expo-document-picker";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { useTheme, useTranslation } from "@/context/UserContext";
import EventSource from "react-native-sse";
import { apiFetch, API_BASE, getAccessToken } from "@/services/api";

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
  sources?: {
    id: string;
    law_name: string;
    article_number: string;
    title: string;
    score: number;
  }[];
  statuses?: string[];
}

interface Conversation {
  id: string;
  title: string;
  preview: string;
  date: string;
  messages: Message[];
}

// ─── API Configuration ────────────────────────────────────────────────────────
// Change this IP to your local machine IP if testing on a physical device.
// Emulators can usually use 10.0.2.2.
const API_URL        = `${API_BASE}/chat`;
const API_URL_STREAM = `${API_BASE}/api/chat/stream`;
const REQUEST_TIMEOUT_MS = 300_000; // 5 minutes — covers worst-case CPU inference on qwen2:7b

// Promise-based timeout (works with Expo's whatwg-fetch polyfill on iOS)
function withTimeout<T>(promise: Promise<T>, ms: number, timeoutMsg: string): Promise<T> {
  return Promise.race([
    promise,
    new Promise<T>((_, reject) =>
      setTimeout(() => reject(new Error(timeoutMsg)), ms)
    ),
  ]);
}

async function getBackendResponse(question: string, history: { role: string; content: string }[], ragType: string) {
  try {
    const fetchPromise = apiFetch("/chat", {
      method: "POST",
      body: JSON.stringify({
        question: question,
        message: question, // alias for local Graph RAG compatibility
        top_k: 7,
        rag_type: ragType,
        history: history,
      }),
    });

    const res = await withTimeout(fetchPromise, REQUEST_TIMEOUT_MS, "TIMEOUT");

    if (!res.ok) {
      throw new Error(`API returned ${res.status}`);
    }

    const data = await res.json();
    return {
      answer: data.answer,
      sources: data.sources || [],
    };
  } catch (error: any) {
    console.error("Backend Error:", error);
    if (error?.message === "TIMEOUT") {
      return {
        answer: "⏳ انتهت مهلة الانتظار (3 دقائق). النموذج يعمل على المعالج وقد يحتاج إلى وقت أطول. يرجى المحاولة مرة أخرى.",
        sources: [],
      };
    }
    return {
      answer: "❌ عذراً، تعذر الاتصال بالخادم. يرجى التحقق من تشغيل الخادم واتصالك بالشبكة.",
      sources: []
    };
  }
}

async function streamBackendResponse(
  question: string,
  history: { role: string; content: string }[],
  onEvent: (event: { type: string; message?: string; content?: string; sources?: any[] }) => void,
  ragType: string = "agentic",
): Promise<void> {
  // EventSource cannot send Authorization headers — pass token as query param instead.
  const token = await getAccessToken();
  const url = token
    ? `${API_URL_STREAM}?token=${encodeURIComponent(token)}`
    : API_URL_STREAM;

  return new Promise((resolve, reject) => {
    const es = new EventSource(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: question,
        rag_type: ragType,
        history: history,
      }),
    });

    es.addEventListener("message", (event) => {
      if (event.data) {
        try {
          const parsed = JSON.parse(event.data || "{}");
          onEvent(parsed);
          if (parsed.type === "done" || parsed.type === "error") {
            es.removeAllEventListeners();
            es.close();
            if (parsed.type === "error") reject(new Error(parsed.message));
            else resolve();
          }
        } catch (e) {
          // Ignore parsing errors for incomplete chunks
        }
      }
    });

    es.addEventListener("error", (err) => {
      console.log("SSE Connection Error:", err);
      es.removeAllEventListeners();
      es.close();
      reject(err);
    });
  });
}


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

function MessageBubble({
  msg, theme, onLongPress,
}: {
  msg: Message;
  theme: any;
  onLongPress: (msg: Message) => void;
}) {
  const isUser = msg.sender === "user";
  const bgColor = isUser ? theme.card : theme.primary;
  const textColor = isUser ? theme.text : "#FFFFFF";

  function formatText(text: string) {
    const parts = text.split(/(\*\*[^*]+\*\*)/g);
    return parts.map((part, i) => {
      if (part.startsWith("**") && part.endsWith("**")) {
        return <Text key={i} style={{ fontFamily: "inter-semibold" }}>{part.slice(2, -2)}</Text>;
      }
      return <Text key={i}>{part}</Text>;
    });
  }

  return (
    <TouchableOpacity
      activeOpacity={0.85}
      onLongPress={() => onLongPress(msg)}
      delayLongPress={400}
      style={{ alignSelf: isUser ? "flex-end" : "flex-start", maxWidth: "82%", marginVertical: 5, marginHorizontal: 14 }}
    >
      <View style={{
        backgroundColor: bgColor, borderRadius: 18,
        borderBottomRightRadius: isUser ? 4 : 18,
        borderBottomLeftRadius: isUser ? 18 : 4,
        paddingHorizontal: 14, paddingVertical: 10,
        shadowColor: theme.shadow, shadowOffset: { width: 0, height: 1 }, shadowOpacity: 0.07, elevation: 2,
      }}>
        {msg.statuses && msg.statuses.length > 0 && (
          <View style={{
            marginBottom: msg.text ? 8 : 0, 
            paddingBottom: msg.text ? 8 : 0,
            borderBottomWidth: msg.text ? 0.5 : 0,
            borderBottomColor: "rgba(255,255,255,0.2)"
          }}>
            {msg.statuses.map((status, i) => (
              <Text key={`s-${i}`} style={{ fontSize: 12, color: "rgba(255,255,255,0.8)", fontStyle: "italic", marginBottom: 3 }}>
                {status}
              </Text>
            ))}
          </View>
        )}
        {msg.type === "text" && msg.text !== undefined && (
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
        {msg.sources && msg.sources.length > 0 && (
          <View style={{ marginTop: 10, paddingTop: 10, borderTopWidth: 1, borderTopColor: isUser ? theme.border : "rgba(255,255,255,0.2)" }}>
            <Text style={{ fontSize: 12, fontFamily: "inter-semibold", color: textColor, marginBottom: 4 }}>المصادر القانونية:</Text>
            {msg.sources.map((src, i) => (
              <View key={i} style={{ marginBottom: 4 }}>
                <Text style={{ fontSize: 11, color: textColor, opacity: 0.9 }}>
                  • {src.law_name} - {src.article_number && `المادة ${src.article_number}`}
                </Text>
              </View>
            ))}
          </View>
        )}
      </View>
      <Text style={{ fontSize: 10, color: theme.textMuted, marginTop: 3, textAlign: isUser ? "right" : "left", paddingHorizontal: 4 }}>
        {new Date(msg.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
      </Text>
    </TouchableOpacity>
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
  const [ragMode, setRagMode] = useState("hybrid");
  const [isInputFocused, setIsInputFocused] = useState(false);

  // Copy/Edit action state
  const [actionTarget, setActionTarget] = useState<Message | null>(null);
  const [editingMessageId, setEditingMessageId] = useState<string | null>(null);
  const [editingText, setEditingText] = useState("");

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
  const emptyStateOpacity = useRef(new Animated.Value(1)).current;

  // Refs
  const flatListRef = useRef<FlatList>(null);
  const sidebarAnim = useRef(new Animated.Value(-320)).current;
  const aiMessages = useRef<{ role: string; content: string }[]>([]);

  // ── Load history ──────────────────────────────────────────────────────────

  useEffect(() => {
    loadConversations();
  }, []);

  useEffect(() => {
    Animated.timing(emptyStateOpacity, {
      toValue: isInputFocused ? 0 : 1,
      duration: 250,
      useNativeDriver: true,
    }).start();
  }, [isInputFocused]);

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

  // ── Copy/Edit handlers ────────────────────────────────────────────────────

  const handleLongPress = (msg: Message) => {
    setActionTarget(msg);
  };

  const handleCopy = async () => {
    if (actionTarget?.text) {
      await Clipboard.setStringAsync(actionTarget.text);
    }
    setActionTarget(null);
  };

  const handleEditStart = () => {
    if (!actionTarget || actionTarget.sender !== "user") return;
    setEditingMessageId(actionTarget.id);
    setEditingText(actionTarget.text || "");
    setActionTarget(null);
  };

  const handleEditSubmit = useCallback(async () => {
    if (!editingMessageId || !editingText.trim()) return;

    // Find the index of the edited message
    const msgIndex = messages.findIndex(m => m.id === editingMessageId);
    if (msgIndex === -1) return;

    // Replace the user message and remove everything after it (including bot reply)
    const updatedMsg: Message = {
      ...messages[msgIndex],
      text: editingText.trim(),
      timestamp: new Date(),
    };
    const trimmedMsgs = [...messages.slice(0, msgIndex), updatedMsg];
    setMessages(trimmedMsgs);
    setEditingMessageId(null);
    setEditingText("");

    // Rebuild AI history up to this message
    aiMessages.current = trimmedMsgs
      .filter(m => m.type === "text" && m.text)
      .map(m => ({ role: m.sender === "user" ? "user" : "assistant", content: m.text! }));

    // Re-send the edited message
    setIsTyping(true);
    setTimeout(() => flatListRef.current?.scrollToEnd({ animated: true }), 100);
    try {
      const historyToSend = aiMessages.current.slice(0, -1);
      
      if (ragMode === "agentic") {
        const botMsgId = makeId();
        const initialBotMsg: Message = {
          id: botMsgId, type: "text", sender: "bot", timestamp: new Date(), text: "", statuses: []
        };
        setMessages(prev => [...prev, initialBotMsg]);
        setIsTyping(false);
        let finalText = "";
        
        await streamBackendResponse(editingText.trim(), historyToSend, (event) => {
          setMessages(prevMsgs => {
            const msgIndex = prevMsgs.findIndex(m => m.id === botMsgId);
            if (msgIndex === -1) return prevMsgs;
            const msg = { ...prevMsgs[msgIndex] };
            if (event.type === "status" && event.message) {
              msg.statuses = [...(msg.statuses || []), event.message];
            } else if (event.type === "token" && event.content) {
              msg.text = (msg.text || "") + event.content;
              finalText = msg.text;
            } else if (event.type === "sources" && event.sources) {
              msg.sources = event.sources;
            } else if (event.type === "done" && event.sources) {
              msg.sources = event.sources;
            }
            const newMsgs = [...prevMsgs];
            newMsgs[msgIndex] = msg;
            return newMsgs;
          });
        });
        
        aiMessages.current.push({ role: "assistant", content: finalText });
        setTimeout(() => flatListRef.current?.scrollToEnd({ animated: true }), 100);
        setMessages(curr => { saveCurrentConversation(curr); return curr; });
      } else {
        const response = await getBackendResponse(editingText.trim(), historyToSend, ragMode);
        const botMsg: Message = {
          id: makeId(), type: "text", sender: "bot",
          timestamp: new Date(), text: response.answer, sources: response.sources,
        };
        const finalMsgs = [...trimmedMsgs, botMsg];
        setMessages(finalMsgs);
        aiMessages.current.push({ role: "assistant", content: response.answer });
        setTimeout(() => flatListRef.current?.scrollToEnd({ animated: true }), 100);
        await saveCurrentConversation(finalMsgs);
      }
    } catch {
      const errMsg: Message = { id: makeId(), type: "text", sender: "bot", timestamp: new Date(), text: "❌ Une erreur est survenue. Veuillez réessayer." };
      setMessages(prev => {
        const emptyBot = prev.find(m => m.sender === "bot" && m.text === "");
        if (emptyBot) return prev.map(m => m.id === emptyBot.id ? { ...m, text: errMsg.text } : m);
        return [...prev, errMsg];
      });
    } finally {
      setIsTyping(false);
    }
  }, [editingMessageId, editingText, messages, ragMode]);

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
      const historyToSend = hasText ? aiMessages.current.slice(0, -1) : aiMessages.current;
      
      if (ragMode === "agentic") {
        const botMsgId = makeId();
        const initialBotMsg: Message = {
          id: botMsgId, type: "text", sender: "bot", timestamp: new Date(), text: "", statuses: []
        };
        setMessages(prev => [...prev, initialBotMsg]);
        setIsTyping(false);
        let finalText = "";
        
        await streamBackendResponse(hasText ? text.trim() : aiContent, historyToSend, (event) => {
          setMessages(prevMsgs => {
            const msgIndex = prevMsgs.findIndex(m => m.id === botMsgId);
            
            if (msgIndex === -1) {
              // The ultimate safeguard: if the token arrives before the initial bubble was 
              // committed, create the bubble now with the first event included.
              const newMsg: Message = { 
                id: botMsgId, 
                type: "text", 
                sender: "bot", 
                timestamp: new Date(), 
                text: event.type === "token" ? event.content || "" : "", 
                statuses: event.type === "status" && event.message ? [event.message] : [],
                sources: event.type === "sources" && event.sources ? event.sources : undefined
              };
              if (event.type === "token" && event.content) finalText += event.content;
              return [...prevMsgs, newMsg];
            }

            const msg = { ...prevMsgs[msgIndex] };
            if (event.type === "status" && event.message) {
              msg.statuses = [...(msg.statuses || []), event.message];
            } else if (event.type === "token" && event.content) {
              msg.text = (msg.text || "") + event.content;
              finalText = msg.text;
            } else if (event.type === "sources" && event.sources) {
              msg.sources = event.sources;
            } else if (event.type === "done" && event.sources) {
              msg.sources = event.sources;
            }
            
            const newMsgs = [...prevMsgs];
            newMsgs[msgIndex] = msg;
            return newMsgs;
          });
        });
        
        aiMessages.current.push({ role: "assistant", content: finalText });
        setTimeout(() => flatListRef.current?.scrollToEnd({ animated: true }), 100);
        setMessages(curr => { saveCurrentConversation(curr); return curr; });
      } else {
        const response = await getBackendResponse(hasText ? text.trim() : aiContent, historyToSend, ragMode);
        const botMsg: Message = { 
          id: makeId(), 
          type: "text", 
          sender: "bot", 
          timestamp: new Date(), 
          text: response.answer,
          sources: response.sources 
        };
        const finalMsgs = [...updated, botMsg];
        setMessages(finalMsgs);
        aiMessages.current.push({ role: "assistant", content: response.answer });
        setTimeout(() => flatListRef.current?.scrollToEnd({ animated: true }), 100);
        await saveCurrentConversation(finalMsgs);
      }
    } catch {
      const errMsg: Message = { id: makeId(), type: "text", sender: "bot", timestamp: new Date(), text: "❌ Une erreur est survenue. Veuillez réessayer." };
      setMessages(prev => {
        const emptyBot = prev.find(m => m.sender === "bot" && m.text === "");
        if (emptyBot) return prev.map(m => m.id === emptyBot.id ? { ...m, text: errMsg.text } : m);
        return [...prev, errMsg];
      });
    } finally {
      setIsTyping(false);
    }
  }, [messages, pendingAttachment, ragMode]);

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
        const response = await getBackendResponse("[Message vocal envoyé]", aiMessages.current, ragMode);
        const botMsg: Message = { 
            id: makeId(), 
            type: "text", 
            sender: "bot", 
            timestamp: new Date(), 
            text: response.answer,
            sources: response.sources
        };
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

  const nextRAGMode = () => {
    const modes = ["hybrid", "agentic", "graph_local", "graph", "bm25", "standard"];
    const currIdx = modes.indexOf(ragMode);
    setRagMode(modes[(currIdx + 1) % modes.length]);
  };

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <TouchableWithoutFeedback onPress={Keyboard.dismiss}>
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
        behavior={Platform.OS === "ios" ? "padding" : undefined}
        style={{ flex: 1 }}
        keyboardVerticalOffset={0}
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
            <TouchableOpacity onPress={nextRAGMode} style={{ marginTop: 2, backgroundColor: theme.primary + "20", paddingHorizontal: 8, paddingVertical: 2, borderRadius: 10 }}>
              <Text style={{ fontSize: 10, fontFamily: "inter-semibold", color: theme.primary }}>
                Mode: {ragMode.toUpperCase()}
              </Text>
            </TouchableOpacity>
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
          <Animated.View style={{ flex: 1, alignItems: "center", justifyContent: "center", padding: 32, opacity: emptyStateOpacity }} pointerEvents={isInputFocused ? "none" : "auto"}>
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
          </Animated.View>
        )}

        {/* ── Messages ── */}
        {messages.length > 0 && (
          <FlatList
            ref={flatListRef}
            data={messages}
            keyExtractor={m => m.id}
            renderItem={({ item }) => (
              editingMessageId === item.id ? (
                // Inline edit mode
                <View style={{ marginHorizontal: 14, marginVertical: 5, alignSelf: "flex-end", maxWidth: "82%" }}>
                  <View style={{ backgroundColor: theme.card, borderRadius: 18, borderBottomRightRadius: 4, paddingHorizontal: 12, paddingVertical: 8, borderWidth: 1.5, borderColor: theme.primary }}>
                    <TextInput
                      value={editingText}
                      onChangeText={setEditingText}
                      autoFocus
                      multiline
                      style={{ fontSize: 15, color: theme.text, minHeight: 36, maxHeight: 120 }}
                    />
                    <View style={{ flexDirection: "row", justifyContent: "flex-end", gap: 10, marginTop: 8 }}>
                      <TouchableOpacity onPress={() => { setEditingMessageId(null); setEditingText(""); }} style={{ paddingHorizontal: 14, paddingVertical: 6, borderRadius: 10, backgroundColor: theme.pillBg }}>
                        <Text style={{ fontSize: 13, color: theme.textSecondary }}>إلغاء</Text>
                      </TouchableOpacity>
                      <TouchableOpacity onPress={handleEditSubmit} style={{ paddingHorizontal: 14, paddingVertical: 6, borderRadius: 10, backgroundColor: theme.primary }}>
                        <Text style={{ fontSize: 13, color: "#fff", fontFamily: "inter-semibold" }}>إرسال ✓</Text>
                      </TouchableOpacity>
                    </View>
                  </View>
                </View>
              ) : (
                <MessageBubble msg={item} theme={theme} onLongPress={handleLongPress} />
              )
            )}
            contentContainerStyle={{ paddingVertical: 12, paddingBottom: 6 }}
            showsVerticalScrollIndicator={false}
            onContentSizeChange={() => flatListRef.current?.scrollToEnd({ animated: true })}
            keyboardShouldPersistTaps="handled"
            keyboardDismissMode="on-drag"
            ListFooterComponent={isTyping ? (
              <View style={{ alignSelf: "flex-start", marginHorizontal: 16, marginVertical: 6 }}>
                <View style={{ backgroundColor: theme.primary, borderRadius: 20, borderBottomLeftRadius: 6, paddingHorizontal: 16, paddingVertical: 12 }}>
                  <ActivityIndicator size="small" color="rgba(255,255,255,0.9)" />
                  <Text style={{ fontSize: 11, color: "rgba(255,255,255,0.75)", marginTop: 6, textAlign: "center" }}>
                    {ragMode === "graph_local" ? "🤖 qwen2 يفكر…" : "جاري التفكير…"}
                  </Text>
                </View>
              </View>
            ) : null}
          />
        )}
        {messages.length === 0 && isTyping && <TypingIndicator theme={theme} />}

        {/* ── Voice Recording Overlay ── */}
        {isRecording && (
          <View style={{
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
          <View style={{ backgroundColor: theme.headerBg }}>

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
                  onFocus={() => setIsInputFocused(true)}
                  onBlur={() => setIsInputFocused(false)}
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

      {/* ── Copy / Edit Action Modal ── */}
      <Modal
        visible={actionTarget !== null}
        transparent
        animationType="fade"
        onRequestClose={() => setActionTarget(null)}
        statusBarTranslucent
      >
        <TouchableWithoutFeedback onPress={() => setActionTarget(null)}>
          <View style={{ flex: 1, backgroundColor: "rgba(0,0,0,0.5)", justifyContent: "center", alignItems: "center" }}>
            <TouchableWithoutFeedback>
              <View style={{
                backgroundColor: theme.card, borderRadius: 20,
                padding: 8, minWidth: 220,
                shadowColor: "#000", shadowOffset: { width: 0, height: 8 }, shadowOpacity: 0.25, elevation: 24,
              }}>
                {/* Preview of the message text */}
                {actionTarget?.text && (
                  <Text numberOfLines={3} style={{ fontSize: 12, color: theme.textMuted, paddingHorizontal: 16, paddingVertical: 10, borderBottomWidth: 1, borderBottomColor: theme.divider }}>
                    {actionTarget.text.slice(0, 120)}{actionTarget.text.length > 120 ? "…" : ""}
                  </Text>
                )}
                {/* Copy button */}
                <TouchableOpacity
                  onPress={handleCopy}
                  style={{ flexDirection: "row", alignItems: "center", paddingHorizontal: 18, paddingVertical: 14, gap: 14 }}
                >
                  <Ionicons name="copy-outline" size={20} color={theme.text} />
                  <Text style={{ fontSize: 15, color: theme.text, fontFamily: "inter-regular" }}>نسخ</Text>
                </TouchableOpacity>
                {/* Edit button — only for user messages */}
                {actionTarget?.sender === "user" && actionTarget?.type === "text" && (
                  <>
                    <View style={{ height: 1, backgroundColor: theme.divider, marginHorizontal: 16 }} />
                    <TouchableOpacity
                      onPress={handleEditStart}
                      style={{ flexDirection: "row", alignItems: "center", paddingHorizontal: 18, paddingVertical: 14, gap: 14 }}
                    >
                      <Ionicons name="create-outline" size={20} color={theme.primary} />
                      <Text style={{ fontSize: 15, color: theme.primary, fontFamily: "inter-regular" }}>تعديل الرسالة</Text>
                    </TouchableOpacity>
                  </>
                )}
                {/* Cancel */}
                <View style={{ height: 1, backgroundColor: theme.divider, marginHorizontal: 16 }} />
                <TouchableOpacity
                  onPress={() => setActionTarget(null)}
                  style={{ paddingHorizontal: 18, paddingVertical: 14, alignItems: "center" }}
                >
                  <Text style={{ fontSize: 14, color: theme.textSecondary }}>إلغاء</Text>
                </TouchableOpacity>
              </View>
            </TouchableWithoutFeedback>
          </View>
        </TouchableWithoutFeedback>
      </Modal>
      </View>
    </TouchableWithoutFeedback>
  );
}
