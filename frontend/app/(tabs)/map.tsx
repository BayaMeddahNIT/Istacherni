import React, { useState, useEffect, useRef } from 'react';
import { View, Text, Dimensions, TouchableOpacity, Alert, Platform } from 'react-native';
import MapView, { Marker, Callout } from 'react-native-maps';
import * as Location from 'expo-location';
import { Ionicons } from '@expo/vector-icons';
import { useTheme, useTranslation } from '@/context/UserContext';

const { width, height } = Dimensions.get('window');

const LAWYERS = [
  { id: '1', name: 'Maître Youssef', type: 'Droit Pénal',        latitude: 36.7525, longitude: 3.04197, rating: 4.8 },
  { id: '2', name: 'Maître Sarah',   type: 'Droit de la Famille', latitude: 36.7645, longitude: 3.0565,  rating: 4.9 },
  { id: '3', name: 'Maître Amine',   type: 'Droit des Affaires',  latitude: 36.7565, longitude: 3.0305,  rating: 4.5 },
  { id: '4', name: 'Maître Meriem',  type: 'Droit du Travail',    latitude: 36.7425, longitude: 3.0555,  rating: 4.7 },
  { id: '5', name: 'Maître Karim',   type: 'Droit Immobilier',    latitude: 36.7725, longitude: 3.0455,  rating: 4.6 },
];

const getDistanceKm = (lat1: number, lon1: number, lat2: number, lon2: number) => {
  const R = 6371;
  const dLat = ((lat2 - lat1) * Math.PI) / 180;
  const dLon = ((lon2 - lon1) * Math.PI) / 180;
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos((lat1 * Math.PI) / 180) * Math.cos((lat2 * Math.PI) / 180) * Math.sin(dLon / 2) ** 2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
};

export default function MapScreen() {
  const theme = useTheme();
  const { t, isRTL } = useTranslation();

  const [location, setLocation] = useState<Location.LocationObject | null>(null);
  const [nearestLawyer, setNearestLawyer] = useState<any>(null);
  const mapRef = useRef<MapView>(null);

  useEffect(() => {
    (async () => {
      const { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert(t('permissionDenied'), t('locationPermDenied'));
        return;
      }
      const userLocation = await Location.getCurrentPositionAsync({});
      setLocation(userLocation);

      if (mapRef.current) {
        mapRef.current.animateToRegion({
          latitude: userLocation.coords.latitude,
          longitude: userLocation.coords.longitude,
          latitudeDelta: 0.05,
          longitudeDelta: 0.05,
        });
      }

      let nearest: any = null;
      let minDist = Infinity;
      LAWYERS.forEach(l => {
        const d = getDistanceKm(
          userLocation.coords.latitude, userLocation.coords.longitude,
          l.latitude, l.longitude
        );
        if (d < minDist) { minDist = d; nearest = l; }
      });
      setNearestLawyer(nearest);
    })();
  }, []);

  const focusUser = () => {
    if (location && mapRef.current) {
      mapRef.current.animateToRegion({
        latitude: location.coords.latitude,
        longitude: location.coords.longitude,
        latitudeDelta: 0.05,
        longitudeDelta: 0.05,
      });
    }
  };

  return (
    <View style={{ flex: 1, backgroundColor: theme.background }}>
      <MapView
        ref={mapRef}
        style={{ flex: 1, width, height }}
        initialRegion={{ latitude: 36.7525, longitude: 3.04197, latitudeDelta: 0.0922, longitudeDelta: 0.0421 }}
        showsUserLocation
        showsMyLocationButton={false}
      >
        {LAWYERS.map(lawyer => {
          const isNearest = nearestLawyer?.id === lawyer.id;
          return (
            <Marker
              key={lawyer.id}
              coordinate={{ latitude: lawyer.latitude, longitude: lawyer.longitude }}
              pinColor={isNearest ? '#E04545' : theme.primary}
            >
              <Callout onPress={() => Alert.alert(lawyer.name, `${lawyer.type}\n⭐ ${lawyer.rating}`)}>
                <View style={{ width: 160, padding: 8 }}>
                  <Text style={{ fontFamily: 'inter-semibold', fontSize: 15, color: '#1A1A1A' }}>
                    {lawyer.name}
                  </Text>
                  <Text style={{ fontFamily: 'inter-regular', fontSize: 13, color: '#807261', marginTop: 2 }}>
                    {lawyer.type}
                  </Text>
                  <View style={{ flexDirection: 'row', alignItems: 'center', marginTop: 4 }}>
                    <Ionicons name="star" size={12} color="#F2C94C" />
                    <Text style={{ fontFamily: 'inter-medium', fontSize: 12, color: '#1A1A1A', marginLeft: 4 }}>
                      {lawyer.rating}
                    </Text>
                  </View>
                  {isNearest && (
                    <View style={{
                      backgroundColor: 'rgba(224,69,69,0.1)', borderRadius: 6,
                      paddingHorizontal: 8, paddingVertical: 4, marginTop: 8, alignSelf: 'flex-start',
                    }}>
                      <Text style={{ fontFamily: 'inter-semibold', fontSize: 11, color: '#E04545' }}>
                        {t('nearest')}
                      </Text>
                    </View>
                  )}
                </View>
              </Callout>
            </Marker>
          );
        })}
      </MapView>

      {/* Floating Header */}
      <View style={{
        position: 'absolute',
        top: Platform.OS === 'ios' ? 60 : 44,
        left: 20, right: 20,
        backgroundColor: theme.card,
        padding: 16, borderRadius: 20,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.13, shadowRadius: 10, elevation: 8,
        borderWidth: 1, borderColor: theme.border,
      }}>
        <Text style={{
          fontSize: 18, fontFamily: 'inter-semibold',
          color: theme.text, textAlign: isRTL ? 'right' : 'left',
        }}>
          {t('findLawyer')}
        </Text>
        <Text style={{
          fontSize: 13, color: theme.textSecondary,
          marginTop: 4, textAlign: isRTL ? 'right' : 'left',
        }}>
          {t('nearbyExperts')}
        </Text>
      </View>

      {/* Locate Me Button */}
      <TouchableOpacity
        activeOpacity={0.8}
        onPress={focusUser}
        style={{
          position: 'absolute', bottom: 30, right: 20,
          backgroundColor: theme.card,
          width: 56, height: 56, borderRadius: 28,
          alignItems: 'center', justifyContent: 'center',
          shadowColor: '#000',
          shadowOffset: { width: 0, height: 4 },
          shadowOpacity: 0.15, shadowRadius: 10, elevation: 8,
          borderWidth: 1, borderColor: theme.border,
        }}
      >
        <Ionicons name="locate" size={24} color={theme.primary} />
      </TouchableOpacity>
    </View>
  );
}
