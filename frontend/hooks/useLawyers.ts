/**
 * useLawyers.ts
 * -------------
 * Convenience hook for consuming LawyerContext inside any screen or component.
 *
 * Usage:
 *   const { lawyers, loading, filters, updateFilters } = useLawyers();
 */

import { useLawyerContext } from "@/context/LawyerContext";

export function useLawyers() {
  return useLawyerContext();
}
