import { LatLng, Trip } from '../types';
import { RegionState } from './state';

function randomInt(min: number, max: number): number {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

// ETA service (stubbed, pretends to call Maps API)
export function estimateEta(origin: LatLng, destination: LatLng): number {
  const dLat = Math.abs(destination.lat - origin.lat);
  const dLng = Math.abs(destination.lng - origin.lng);
  const base = (dLat + dLng) * 100;
  const noise = randomInt(3, 10);
  return Math.max(3, Math.round(base + noise));
}

// Pricing service (stubbed, pretends to use surge and fare rules)
export function calculatePriceCents(origin: LatLng, destination: LatLng): number {
  const dLat = Math.abs(destination.lat - origin.lat);
  const dLng = Math.abs(destination.lng - origin.lng);
  const distanceFactor = (dLat + dLng) * 10000;
  const baseFare = 500; // $5.00
  const surgeMultiplier = 1 + Math.random() * 0.5;
  return Math.round((baseFare + distanceFactor) * surgeMultiplier);
}

// Dispatch service (uses GeoStore + fairness tracking)
export function assignDriver(state: RegionState, trip: Trip): Trip | undefined {
  const nearby = state.geo.findNearby(state.region, trip.origin.lat, trip.origin.lng, 10);
  if (nearby.length === 0) {
    return state.trips.update(trip.id, { status: 'requested' }); // stay requested, no driver yet
  }

  // DISPATCH FAIRNESS: prefer drivers with fewer recent trips
  // Score = distance * (1 + tripCount * fairnessWeight)
  // Lower score = better (closer + fewer trips = preferred)
  const fairnessWeight = 0.3; // how much to penalize busy drivers
  const candidates = nearby.map((driver) => {
    const dLat = driver.position.lat - trip.origin.lat;
    const dLng = driver.position.lng - trip.origin.lng;
    const distance = Math.sqrt(dLat * dLat + dLng * dLng);
    const tripCount = state.driverFairness.getCount(driver.driverId);
    const score = distance * (1 + tripCount * fairnessWeight);
    return { driver, score, distance, tripCount };
  });

  // Sort by score (best = closest + least busy)
  candidates.sort((a, b) => a.score - b.score);
  const selected = candidates[0].driver;

  // Increment trip count for fairness tracking
  state.driverFairness.increment(selected.driverId);

  // Periodically decay counts (simulate time window reset)
  if (Math.random() < 0.1) {
    state.driverFairness.reset();
  }

  const updated = state.trips.update(trip.id, {
    driverId: selected.driverId,
    status: 'assigned',
  });

  if (updated) {
    state.notifications.add('trip_assigned', {
      tripId: updated.id,
      driverId: updated.driverId,
      riderId: updated.riderId,
      region: updated.region,
      fairness: {
        selectedDriverTripCount: candidates[0].tripCount,
        consideredDrivers: candidates.length,
      },
    });
  }

  return updated;
}


