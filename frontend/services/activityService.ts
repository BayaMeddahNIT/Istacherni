import AsyncStorage from "@react-native-async-storage/async-storage";

export type RecentActivity = {
  id: string;
  type: "search" | "document" | "gazette";
  title: string;
  category?: string;
  query?: string;
  documentId?: string;
  url?: string;
  timestamp: number;
};

const STORAGE_KEY = "istacherni_recent_activity";
const MAX_ITEMS = 10;

export const activityService = {
  async addActivity(activity: Omit<RecentActivity, "id" | "timestamp">) {
    try {
      const existing = await this.getActivities();
      
      // Prevent consecutive duplicates of the same title/type
      if (existing.length > 0 && existing[0].title === activity.title && existing[0].type === activity.type) {
        return;
      }

      const newActivity: RecentActivity = {
        ...activity,
        title: String(activity.title || "").trim(),
        query: activity.query ? String(activity.query).trim() : undefined,
        id: Math.random().toString(36).substring(7),
        timestamp: Date.now(),
      };

      const updated = [newActivity, ...existing.filter(a => a.title !== activity.title || a.type !== activity.type)].slice(0, MAX_ITEMS);
      await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
    } catch (error) {
      console.error("Error adding activity:", error);
    }
  },

  async getActivities(): Promise<RecentActivity[]> {
    try {
      const data = await AsyncStorage.getItem(STORAGE_KEY);
      return data ? JSON.parse(data) : [];
    } catch (error) {
      console.error("Error getting activities:", error);
      return [];
    }
  },

  async clearActivities() {
    try {
      await AsyncStorage.removeItem(STORAGE_KEY);
    } catch (error) {
      console.error("Error clearing activities:", error);
    }
  }
};
