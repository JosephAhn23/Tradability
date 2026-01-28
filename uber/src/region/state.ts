import { DriverLocation, RegionCode, Trip } from '../types';

/**
 * Simple in-memory stores that simulate region-local Postgres and Redis GEO.
 * In a real system, these would be replaced by proper data layers.
 */

class TripStore {
  private trips = new Map<string, Trip>();

  create(trip: Trip): Trip {
    this.trips.set(trip.id, trip);
    return trip;
  }

  get(id: string): Trip | undefined {
    return this.trips.get(id);
  }

  update(id: string, updates: Partial<Trip>): Trip | undefined {
    const existing = this.trips.get(id);
    if (!existing) return undefined;
    const updated: Trip = {
      ...existing,
      ...updates,
      updatedAt: new Date().toISOString(),
    };
    this.trips.set(id, updated);
    return updated;
  }
}

class GeoStore {
  private locations = new Map<string, DriverLocation>();

  upsert(location: DriverLocation): DriverLocation {
    this.locations.set(location.driverId, location);
    return location;
  }

  /**
   * Very naive "nearby" lookup using Euclidean distance in lat/lng space.
   * In real life, you'd use Redis GEO or a dedicated geo index.
   */
  findNearby(region: RegionCode, originLat: number, originLng: number, limit = 5): DriverLocation[] {
    const all = Array.from(this.locations.values()).filter((l) => l.region === region);
    const withDistance = all.map((loc) => {
      const dLat = loc.position.lat - originLat;
      const dLng = loc.position.lng - originLng;
      const distance = Math.sqrt(dLat * dLat + dLng * dLng);
      return { loc, distance };
    });
    return withDistance
      .sort((a, b) => a.distance - b.distance)
      .slice(0, limit)
      .map((x) => x.loc);
  }
}

/**
 * Tracks driver trip counts for fairness (prevents driver starvation).
 * In a real system, this would be in Redis or a database with TTL.
 */
class DriverFairnessTracker {
  private tripCounts = new Map<string, number>(); // driverId -> trip count

  increment(driverId: string) {
    const current = this.tripCounts.get(driverId) || 0;
    this.tripCounts.set(driverId, current + 1);
  }

  getCount(driverId: string): number {
    return this.tripCounts.get(driverId) || 0;
  }

  /**
   * Reset counts periodically (simulate time window).
   * In production, you'd use a sliding window or TTL.
   */
  reset() {
    // Simple: decay counts over time (simulate hourly reset)
    for (const [driverId, count] of this.tripCounts.entries()) {
      if (count > 0) {
        this.tripCounts.set(driverId, Math.max(0, count - 0.1));
      }
    }
  }
}

class NotificationOutbox {
  private events: Array<{ id: string; type: string; payload: unknown; createdAt: string }> = [];

  add(type: string, payload: unknown) {
    const event = {
      id: `notif_${Date.now()}_${Math.random().toString(36).slice(2)}`,
      type,
      payload,
      createdAt: new Date().toISOString(),
    };
    this.events.push(event);
    // For now, log to console to simulate push/SMS/email.
    // eslint-disable-next-line no-console
    console.log('[NotificationOutbox]', JSON.stringify(event, null, 2));
  }

  list() {
    return this.events;
  }
}

export interface RegionState {
  region: RegionCode;
  trips: TripStore;
  geo: GeoStore;
  notifications: NotificationOutbox;
  driverFairness: DriverFairnessTracker;
}

export function createRegionState(region: RegionCode): RegionState {
  return {
    region,
    trips: new TripStore(),
    geo: new GeoStore(),
    notifications: new NotificationOutbox(),
    driverFairness: new DriverFairnessTracker(),
  };
}


