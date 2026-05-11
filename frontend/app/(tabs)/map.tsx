import React, { useState, useEffect, useRef, useMemo, useCallback, memo } from 'react';
import {
  View, Text, TouchableOpacity, Alert, Platform,
  TextInput, Linking, Animated, StyleSheet, ActivityIndicator, KeyboardAvoidingView,
  TouchableWithoutFeedback, Keyboard,
} from 'react-native';
import MapView, { Marker, Polyline } from 'react-native-maps';
import * as Location from 'expo-location';
import { Ionicons } from '@expo/vector-icons';
import { useTheme, useTranslation } from '@/context/UserContext';
import { useLawyerContext } from '@/context/LawyerContext';
import {
  Lawyer,
  fetchNearbyLawyers,
  fetchLawyerDetails,
  fetchDirections,
  DirectionsResult,
} from '@/services/lawyerService';

// ─── Helpers ──────────────────────────────────────────────────────────────────

const haversineKm = (lat1: number, lon1: number, lat2: number, lon2: number) => {
  const R = 6371;
  const dLat = ((lat2 - lat1) * Math.PI) / 180;
  const dLon = ((lon2 - lon1) * Math.PI) / 180;
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos((lat1 * Math.PI) / 180) *
    Math.cos((lat2 * Math.PI) / 180) *
    Math.sin(dLon / 2) ** 2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
};

const fmtDist = (km: number) => (km < 1 ? `${Math.round(km * 1000)} m` : `${km.toFixed(1)} km`);
const fmtTime = (km: number) => {
  const min = Math.max(1, Math.round((km / 25) * 60));
  return min < 60 ? `${min} min` : `${Math.floor(min / 60)}h${min % 60}min`;
};

const initials = (name: string) =>
  name.replace(/Maître\s*|Cabinet\s*/gi, '').trim().split(/\s+/).slice(0, 2).map(w => w[0]?.toUpperCase() ?? '').join('') || 'AV';

// ─── Custom Marker ─────────────────────────────────────────────────────────────

interface MarkerViewProps { label: string; isSelected: boolean; isNearest: boolean; primaryColor: string; }
const MarkerView = memo(({ label, isSelected, isNearest, primaryColor }: MarkerViewProps) => {
  const bg = isSelected ? '#E04545' : isNearest ? '#C4885C' : primaryColor;
  return (
    // pointerEvents="none" is CRITICAL: prevents this View from
    // swallowing touch events so the parent <Marker onPress> fires.
    <View
      pointerEvents="none"
      style={[styles.markerOuter, { backgroundColor: bg, borderColor: isSelected ? '#fff' : 'rgba(255,255,255,0.75)' }]}
    >
      <Text style={styles.markerText}>{label}</Text>
      <View style={[styles.markerPin, { backgroundColor: bg }]} />
    </View>
  );
});

// ─── Screen ───────────────────────────────────────────────────────────────────

export default function MapScreen() {
  const theme = useTheme();
  const { t } = useTranslation();
  const { setLawyers, setLoading: setCtxLoading, setUserCoords } = useLawyerContext();

  const [location, setLocation] = useState<Location.LocationObject | null>(null);
  const [lawyers, setLocalLawyers] = useState<Lawyer[]>([]);
  const [fromGoogle, setFromGoogle] = useState(false);
  const [mapLoading, setMapLoading] = useState(true);
  const [nearestId, setNearestId] = useState<string | null>(null);
  const [selectedLawyer, setSelectedLawyer] = useState<Lawyer | null>(null);
  const [directions, setDirections] = useState<DirectionsResult | null>(null);
  const [dirLoading, setDirLoading] = useState(false);
  const [navStarted, setNavStarted] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [debouncedQuery, setDebouncedQuery] = useState('');
  const mapRef = useRef<MapView>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const cardAnim = useRef(new Animated.Value(0)).current;

  // ── Debounced search ───────────────────────────────────────────────────────
  const handleSearch = useCallback((text: string) => {
    setSearchQuery(text);
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => setDebouncedQuery(text), 350);
  }, []);

  // ── Filtered markers ───────────────────────────────────────────────────────
  const filteredLawyers = useMemo(() => {
    const q = debouncedQuery.toLowerCase().trim();
    if (!q) return lawyers;
    return lawyers.filter(l =>
      l.name.toLowerCase().includes(q) ||
      l.specialty.toLowerCase().includes(q) ||
      l.city.toLowerCase().includes(q) ||
      l.address.toLowerCase().includes(q)
    );
  }, [lawyers, debouncedQuery]);

  // Tap marker → show info card (NO route yet)
  const handleMarkerPress = useCallback((lawyer: Lawyer) => {
    console.log('🔥 MARKER PRESSED:', lawyer.id, lawyer.name);
    // IMPORTANT: Do NOT put side effects inside setSelectedLawyer updater
    setSelectedLawyer(prev => {
      const next = prev?.id === lawyer.id ? null : lawyer;
      return next;
    });
    setDirections(null);
    setNavStarted(false);
    Animated.spring(cardAnim, { toValue: 1, useNativeDriver: true, tension: 80, friction: 12 }).start();
  }, [cardAnim]);

  const showCard = useCallback((lawyer: Lawyer | null) => {
    if (!lawyer) {
      console.log('[MAP] Card dismissed');
      setSelectedLawyer(null);
      setDirections(null);
      setNavStarted(false);
      Animated.spring(cardAnim, { toValue: 0, useNativeDriver: true, tension: 80, friction: 12 }).start();
    }
  }, [cardAnim]);

  // "Start Navigation" button → fetch real route + fit camera
  const startNavigation = useCallback(async (lawyer: Lawyer) => {
    if (!location || !lawyer.latitude || !lawyer.longitude) {
      openExternalDirections(lawyer);
      return;
    }
    setNavStarted(true);
    setDirLoading(true);
    setDirections(null);
    const result = await fetchDirections(
      location.coords.latitude, location.coords.longitude,
      lawyer.latitude, lawyer.longitude,
    );
    setDirections(result);
    setDirLoading(false);
    if (mapRef.current) {
      const pts = [
        { latitude: location.coords.latitude, longitude: location.coords.longitude },
        { latitude: lawyer.latitude, longitude: lawyer.longitude },
        ...(result?.polyline ?? []),
      ];
      mapRef.current.fitToCoordinates(pts, {
        edgePadding: { top: 130, right: 36, bottom: 280, left: 36 },
        animated: true,
      });
    }
  }, [location]);

  const cardY = cardAnim.interpolate({ inputRange: [0, 1], outputRange: [220, 0] });

  // ── Initial load: location + real Places fetch ─────────────────────────────
  useEffect(() => {
    (async () => {
      setMapLoading(true);
      setCtxLoading(true);

      const { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert(t('permissionDenied'), t('locationPermDenied'));
        setMapLoading(false);
        setCtxLoading(false);
        return;
      }

      const userLoc = await Location.getCurrentPositionAsync({});
      setLocation(userLoc);
      setUserCoords({ latitude: userLoc.coords.latitude, longitude: userLoc.coords.longitude });

      mapRef.current?.animateToRegion({
        latitude: userLoc.coords.latitude,
        longitude: userLoc.coords.longitude,
        latitudeDelta: 0.05,
        longitudeDelta: 0.05,
      });

      // Fetch real lawyers from Google Places
      const { lawyers: fetched, fromGoogle: isGoogle } = await fetchNearbyLawyers(
        userLoc.coords.latitude,
        userLoc.coords.longitude,
      );

      // Sort by distance from user
      const withDist = fetched
        .filter(l => l.latitude && l.longitude)
        .map(l => ({ ...l, _km: haversineKm(userLoc.coords.latitude, userLoc.coords.longitude, l.latitude!, l.longitude!) }))
        .sort((a, b) => a._km - b._km);

      setLocalLawyers(withDist);
      setFromGoogle(isGoogle);
      setNearestId(withDist[0]?.id ?? null);

      // Push to context (consumed by list screen)
      setLawyers(withDist, isGoogle);
      setCtxLoading(false);
      setMapLoading(false);
    })();
  }, []);

  const focusUser = useCallback(() => {
    if (!location) return;
    mapRef.current?.animateToRegion({
      latitude: location.coords.latitude,
      longitude: location.coords.longitude,
      latitudeDelta: 0.04,
      longitudeDelta: 0.04,
    });
  }, [location]);

  const openExternalDirections = useCallback((lawyer: Lawyer) => {
    const { latitude: lat, longitude: lng, name } = lawyer;
    const label = encodeURIComponent(name ?? '');
    const url = Platform.OS === 'ios'
      ? `maps://app?daddr=${lat},${lng}&q=${label}`
      : `https://www.google.com/maps/dir/?api=1&destination=${lat},${lng}`;
    Linking.openURL(url).catch(() =>
      Linking.openURL(`https://www.google.com/maps/dir/?api=1&destination=${lat},${lng}`)
    );
  }, []);

  // On-demand phone fetch for Google lawyers
  const handleCall = useCallback(async (lawyer: Lawyer) => {
    if (lawyer.phone) {
      Linking.openURL('tel:' + lawyer.phone.replace(/\s/g, ''));
      return;
    }
    if (lawyer.source === 'google') {
      const details = await fetchLawyerDetails(lawyer.id);
      if (details.phone) {
        Linking.openURL('tel:' + details.phone.replace(/\s/g, ''));
      } else if (lawyer.mapsUrl) {
        Linking.openURL(lawyer.mapsUrl);
      }
    }
  }, []);

  return (
    <TouchableWithoutFeedback onPress={Keyboard.dismiss} accessible={false}>
      <View style={{ flex: 1, backgroundColor: theme.background }}>
        <MapView
          ref={mapRef}
          style={{ flex: 1 }}
          initialRegion={{ latitude: 36.7525, longitude: 3.04197, latitudeDelta: 0.0922, longitudeDelta: 0.0421 }}
          showsUserLocation
          showsMyLocationButton={false}
        // DO NOT put onPress here on iOS/Apple Maps — it consumes ALL touch
        // events including marker taps. Dismiss via the card's close button instead.
        >
          {/* Route polyline — shadow layer + main */}
          {directions?.polyline && directions.polyline.length > 0 && (
            <>
              <Polyline coordinates={directions.polyline} strokeColor="rgba(66,133,244,0.25)" strokeWidth={10} />
              <Polyline coordinates={directions.polyline} strokeColor="#4285F4" strokeWidth={5} />
            </>
          )}

          {/* Lawyer markers */}
          {filteredLawyers.map(lawyer => {
            if (!lawyer.latitude || !lawyer.longitude) return null;
            const isSelected = selectedLawyer?.id === lawyer.id;
            const isNearest = lawyer.id === nearestId;
            const isDimmed = !!selectedLawyer && !isSelected;
            return (
              <Marker
                key={lawyer.id}
                coordinate={{ latitude: lawyer.latitude, longitude: lawyer.longitude }}
                // tracksViewChanges=true ensures marker responds to touch on iOS
                tracksViewChanges
                // calloutEnabled=false prevents native callout from stealing the gesture
                calloutEnabled={false}
                onPress={() => {
                  console.log('🔥 MARKER CLICKED:', lawyer.id, lawyer.name);
                  handleMarkerPress(lawyer);
                }}
                opacity={isDimmed ? 0.4 : 1}
              >
                <MarkerView
                  label={initials(lawyer.name)}
                  isSelected={isSelected}
                  isNearest={isNearest}
                  primaryColor={theme.primary}
                />
              </Marker>
            );
          })}
        </MapView>

        {/* ── Loading overlay ── */}
        {mapLoading && (
          <View style={[styles.loadingOverlay, { backgroundColor: theme.background + 'CC' }]}>
            <ActivityIndicator size="large" color={theme.primary} />
            <Text style={[styles.loadingText, { color: theme.textSecondary }]}>
              Recherche des avocats proches…
            </Text>
          </View>
        )}

        {/* ── Search bar ── */}
        <View style={[styles.searchWrap, { top: Platform.OS === 'ios' ? 60 : 44 }]} pointerEvents="box-none">
          <View style={[styles.searchBox, { backgroundColor: theme.card, borderColor: theme.border }]}>
            <Ionicons name="search-outline" size={18} color={theme.textMuted} />
            <TextInput
              value={searchQuery}
              onChangeText={handleSearch}
              placeholder="Nom, spécialité, ville…"
              placeholderTextColor={theme.textMuted}
              style={[styles.searchInput, { color: theme.text }]}
              returnKeyType="search"
            />
            {searchQuery.length > 0 && (
              <TouchableOpacity onPress={() => handleSearch('')}>
                <Ionicons name="close-circle" size={18} color={theme.textMuted} />
              </TouchableOpacity>
            )}
          </View>
          {debouncedQuery.length > 0 && (
            <View style={[styles.badge, { backgroundColor: theme.primary }]}>
              <Text style={styles.badgeText}>{filteredLawyers.length} résultat{filteredLawyers.length !== 1 ? 's' : ''}</Text>
            </View>
          )}
          {!fromGoogle && !mapLoading && (
            <View style={[styles.badge, { backgroundColor: '#E67E22' }]}>
              <Ionicons name="warning-outline" size={12} color="#fff" />
              <Text style={styles.badgeText}> Données locales · Activez Places API</Text>
            </View>
          )}
        </View>

        {/* ── Locate me ── */}
        <TouchableOpacity
          activeOpacity={0.85}
          onPress={focusUser}
          style={[styles.locateBtn, { backgroundColor: theme.card, borderColor: theme.border }]}
        >
          <Ionicons name="locate" size={24} color={theme.primary} />
        </TouchableOpacity>

        {/* ── Bottom sheet ── */}
        {selectedLawyer && (
          <Animated.View
            style={[styles.card, { backgroundColor: theme.card, borderColor: theme.border, transform: [{ translateY: cardY }] }]}
          >
            {/* Drag handle */}
            <View style={styles.handle} />

            {/* Close */}
            <TouchableOpacity style={styles.cardClose} onPress={() => showCard(null)}>
              <Ionicons name="close" size={20} color={theme.textMuted} />
            </TouchableOpacity>

            {/* ── Lawyer header ── */}
            <View style={styles.cardHeader}>
              <View style={[styles.avatar, { backgroundColor: theme.primaryLight, borderColor: theme.primaryMedium }]}>
                <Text style={[styles.avatarText, { color: theme.primary }]}>{initials(selectedLawyer.name)}</Text>
              </View>
              <View style={{ flex: 1 }}>
                <Text style={[styles.cardName, { color: theme.text }]} numberOfLines={1}>{selectedLawyer.name}</Text>
                <View style={{ flexDirection: 'row', gap: 6, marginTop: 3, flexWrap: 'wrap' }}>
                  <View style={[styles.pill, { backgroundColor: theme.primaryLight }]}>
                    <Text style={[styles.pillText, { color: theme.primary }]}>{selectedLawyer.specialty}</Text>
                  </View>
                  <View style={[styles.pill, { backgroundColor: selectedLawyer.available ? theme.success + '20' : theme.pillBg }]}>
                    <Text style={[styles.pillText, { color: selectedLawyer.available ? theme.success : theme.textMuted }]}>
                      {selectedLawyer.available ? '● Disponible' : '○ Occupé'}
                    </Text>
                  </View>
                </View>
              </View>
            </View>

            {/* ── Address + haversine distance ── */}
            <View style={[styles.infoRow, { backgroundColor: theme.pillBg }]}>
              <Ionicons name="location-outline" size={14} color={theme.textMuted} />
              <Text style={[styles.infoText, { color: theme.textSecondary }]} numberOfLines={1}>
                {selectedLawyer.address || selectedLawyer.city || 'Algérie'}
              </Text>
              {location && selectedLawyer.latitude ? (
                <View style={[styles.distChip, { backgroundColor: theme.primaryLight }]}>
                  <Ionicons name="navigate-outline" size={11} color={theme.primary} />
                  <Text style={[styles.distChipText, { color: theme.primary }]}>
                    {fmtDist(haversineKm(location.coords.latitude, location.coords.longitude, selectedLawyer.latitude!, selectedLawyer.longitude!))}
                  </Text>
                </View>
              ) : null}
            </View>

            {/* ── ETA bar (shown after navigation starts) ── */}
            {navStarted && (
              <View style={[styles.etaBar, { backgroundColor: '#4285F4' + '18', borderColor: '#4285F4' + '40' }]}>
                {dirLoading ? (
                  <><ActivityIndicator size="small" color="#4285F4" /><Text style={[styles.etaText, { color: '#4285F4' }]}>  Calcul de l'itinéraire…</Text></>
                ) : directions ? (
                  <>
                    <Ionicons name="navigate" size={14} color="#4285F4" />
                    <Text style={[styles.etaText, { color: '#4285F4' }]}>{directions.distanceText}</Text>
                    <View style={{ width: 1, height: 14, backgroundColor: '#4285F440', marginHorizontal: 6 }} />
                    <Ionicons name="time-outline" size={14} color="#4285F4" />
                    <Text style={[styles.etaText, { color: '#4285F4' }]}>{directions.durationText}</Text>
                  </>
                ) : (
                  <><Ionicons name="warning-outline" size={14} color="#E67E22" /><Text style={[styles.etaText, { color: '#E67E22' }]}>  Itinéraire non disponible</Text></>
                )}
              </View>
            )}

            {/* ── Primary: Start Navigation ── */}
            <TouchableOpacity
              style={[styles.navBtn, { backgroundColor: navStarted && directions ? '#1a73e8' : '#4285F4' }]}
              onPress={() => startNavigation(selectedLawyer)}
              activeOpacity={0.85}
            >
              {dirLoading ? (
                <ActivityIndicator size="small" color="#fff" />
              ) : (
                <><Ionicons name={navStarted && directions ? 'navigate' : 'navigate-outline'} size={18} color="#fff" />
                  <Text style={styles.navBtnText}>{navStarted && directions ? 'Ouvrir dans Maps' : 'Démarrer la navigation'}</Text></>
              )}
            </TouchableOpacity>

            {/* ── Secondary actions ── */}
            <View style={styles.actions}>
              <TouchableOpacity style={[styles.btn, { backgroundColor: theme.success + '18', borderWidth: 1, borderColor: theme.success + '35' }]} onPress={() => handleCall(selectedLawyer)}>
                <Ionicons name="call-outline" size={16} color={theme.success} />
                <Text style={[styles.btnText, { color: theme.success }]}>Appeler</Text>
              </TouchableOpacity>
              <TouchableOpacity style={[styles.btn, { backgroundColor: '#25D36614', borderWidth: 1, borderColor: '#25D36635' }]} onPress={() => Linking.openURL('https://wa.me/' + (selectedLawyer.phone || '').replace(/[^0-9]/g, ''))}>
                <Ionicons name="logo-whatsapp" size={16} color="#25D366" />
                <Text style={[styles.btnText, { color: '#25D366' }]}>WhatsApp</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.btn, { backgroundColor: theme.pillBg, borderWidth: 1, borderColor: theme.border }]}
                onPress={() => {
                  const lat = selectedLawyer.latitude;
                  const lng = selectedLawyer.longitude;
                  if (!lat || !lng) {
                    console.warn('[MAP] No coordinates for Maps button, using mapsUrl');
                    if (selectedLawyer.mapsUrl) Linking.openURL(selectedLawyer.mapsUrl);
                    return;
                  }
                  // Use coordinates — always resolves in Google Maps, never "No results"
                  const url = `https://www.google.com/maps/search/?api=1&query=${lat},${lng}`;
                  console.log('[MAP] Opening Google Maps:', url);
                  Linking.openURL(url);
                }}
              >
                <Ionicons name="open-outline" size={16} color={theme.textSecondary} />
                <Text style={[styles.btnText, { color: theme.textSecondary }]}>Maps</Text>
              </TouchableOpacity>
            </View>
          </Animated.View>
        )}
      </View>
    </TouchableWithoutFeedback>
  );
}


// ─── Styles ───────────────────────────────────────────────────────────────────

const styles = StyleSheet.create({
  loadingOverlay: {
    ...StyleSheet.absoluteFillObject, alignItems: 'center', justifyContent: 'center', gap: 14, zIndex: 20,
  },
  loadingText: { fontSize: 14, fontFamily: 'inter-medium' },
  markerOuter: {
    width: 44, height: 44, borderRadius: 22, borderWidth: 3,
    alignItems: 'center', justifyContent: 'center',
    shadowColor: '#000', shadowOffset: { width: 0, height: 3 }, shadowOpacity: 0.3, shadowRadius: 4, elevation: 6,
  },
  markerText: { color: '#fff', fontFamily: 'inter-semibold', fontSize: 13 },
  markerPin: {
    position: 'absolute', bottom: -10, width: 8, height: 8,
    borderRadius: 4, borderWidth: 2, borderColor: '#fff',
  },
  searchWrap: { position: 'absolute', left: 16, right: 16, zIndex: 10 },
  searchBox: {
    flexDirection: 'row', alignItems: 'center', gap: 8,
    borderRadius: 16, borderWidth: 1, paddingHorizontal: 14, height: 50,
    shadowColor: '#000', shadowOffset: { width: 0, height: 3 }, shadowOpacity: 0.15, shadowRadius: 8, elevation: 8,
  },
  searchInput: { flex: 1, fontSize: 14, fontFamily: 'inter-regular' },
  badge: {
    flexDirection: 'row', alignSelf: 'flex-start', marginTop: 8,
    paddingHorizontal: 12, paddingVertical: 5, borderRadius: 12, alignItems: 'center',
  },
  badgeText: { color: '#fff', fontSize: 12, fontFamily: 'inter-semibold' },
  locateBtn: {
    position: 'absolute', bottom: 30, right: 20,
    width: 56, height: 56, borderRadius: 28,
    alignItems: 'center', justifyContent: 'center',
    borderWidth: 1,
    shadowColor: '#000', shadowOffset: { width: 0, height: 4 }, shadowOpacity: 0.15, shadowRadius: 10, elevation: 8,
  },
  card: {
    position: 'absolute', bottom: 100, left: 14, right: 14,
    zIndex: 999, elevation: 20,
    borderRadius: 24, padding: 18, borderWidth: 1,
    shadowColor: '#000', shadowOffset: { width: 0, height: -4 }, shadowOpacity: 0.12, shadowRadius: 12,
  },
  cardClose: { position: 'absolute', top: 14, right: 14, padding: 4 },
  cardHeader: { flexDirection: 'row', alignItems: 'center', gap: 12, marginBottom: 12 },
  avatar: {
    width: 50, height: 50, borderRadius: 16, borderWidth: 2,
    alignItems: 'center', justifyContent: 'center',
  },
  avatarText: { fontSize: 16, fontFamily: 'inter-semibold' },
  cardName: { fontSize: 15, fontFamily: 'inter-semibold', marginBottom: 4 },
  pill: { alignSelf: 'flex-start', borderRadius: 8, paddingHorizontal: 8, paddingVertical: 3 },
  pillText: { fontSize: 11, fontFamily: 'inter-semibold' },
  availBadge: { borderRadius: 10, paddingHorizontal: 8, paddingVertical: 4, alignSelf: 'flex-start' },
  availText: { fontSize: 11, fontFamily: 'inter-semibold' },
  distRow: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'center',
    borderRadius: 14, paddingHorizontal: 12, paddingVertical: 10, marginBottom: 10, minHeight: 44,
  },
  distItem: { flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 5 },
  distText: { fontSize: 12, fontFamily: 'inter-semibold' },
  divider: { width: 1, height: 20, marginHorizontal: 4 },
  addrRow: { flexDirection: 'row', alignItems: 'center', gap: 6, marginBottom: 14 },
  addrText: { fontSize: 12, flex: 1 },
  // Google Maps-style bottom sheet additions
  handle: { width: 36, height: 4, borderRadius: 2, backgroundColor: '#D1D5DB', alignSelf: 'center', marginBottom: 14 },
  infoRow: { flexDirection: 'row', alignItems: 'center', gap: 6, borderRadius: 12, paddingHorizontal: 12, paddingVertical: 9, marginBottom: 12 },
  infoText: { fontSize: 12, flex: 1 },
  distChip: { flexDirection: 'row', alignItems: 'center', gap: 3, borderRadius: 8, paddingHorizontal: 7, paddingVertical: 3 },
  distChipText: { fontSize: 11, fontFamily: 'inter-semibold' },
  etaBar: { flexDirection: 'row', alignItems: 'center', borderRadius: 12, paddingHorizontal: 14, paddingVertical: 10, marginBottom: 12, borderWidth: 1, gap: 4 },
  etaText: { fontSize: 13, fontFamily: 'inter-semibold' },
  navBtn: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8, borderRadius: 16, paddingVertical: 14, marginBottom: 12 },
  navBtnText: { color: '#fff', fontSize: 14, fontFamily: 'inter-semibold' },
  actions: { flexDirection: 'row', gap: 8 },
  btn: {
    flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center',
    gap: 5, paddingVertical: 11, borderRadius: 14,
  },
  btnText: { color: '#fff', fontSize: 12, fontFamily: 'inter-semibold' },
});
