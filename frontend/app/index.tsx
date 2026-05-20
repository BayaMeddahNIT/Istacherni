/**
 * index.tsx — entry point redirect
 * The root _layout.tsx handles the auth guard.
 * This file just renders nothing while the guard runs.
 */
import { useEffect } from "react";
import { router } from "expo-router";

export default function Index() {
  // The root _layout.tsx RootNavigator handles all routing decisions.
  // We render nothing here; it will redirect immediately.
  return null;
}