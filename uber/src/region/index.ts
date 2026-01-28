import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import morgan from 'morgan';
import dotenv from 'dotenv';
import { RegionCode, Trip } from '../types';
import { createRegionState } from './state';
import { assignDriver, calculatePriceCents, estimateEta } from './services';

dotenv.config();

const REGION_ENV = (process.env.REGION || 'A').toUpperCase();
if (!['A', 'B'].includes(REGION_ENV)) {
  // eslint-disable-next-line no-console
  console.error('Invalid REGION env, expected A or B');
  process.exit(1);
}
const REGION = REGION_ENV as RegionCode;

const app = express();
app.use(express.json());
app.use(cors());
app.use(helmet());
app.use(morgan('dev'));

const state = createRegionState(REGION);

// Health
app.get('/health', (_req, res) => {
  res.json({ status: 'ok', region: REGION });
});

// Trip Service: request a trip
app.post('/trip/request', (req, res) => {
  const { riderId, origin, destination } = req.body || {};
  if (!riderId || !origin || !destination) {
    return res.status(400).json({ error: 'riderId, origin, destination are required' });
  }

  const now = new Date().toISOString();
  const id = `trip_${REGION}_${Date.now()}_${Math.random().toString(36).slice(2)}`;
  const etaMinutes = estimateEta(origin, destination);
  const priceCents = calculatePriceCents(origin, destination);

  const trip: Trip = {
    id,
    region: REGION,
    riderId,
    origin,
    destination,
    etaMinutes,
    priceCents,
    currency: 'USD',
    status: 'requested',
    createdAt: now,
    updatedAt: now,
  };

  state.trips.create(trip);

  state.notifications.add('trip_requested', {
    tripId: trip.id,
    riderId: trip.riderId,
    region: trip.region,
  });

  return res.status(201).json(trip);
});

// Trip Service: get trip
app.get('/trip/:id', (req, res) => {
  const trip = state.trips.get(req.params.id);
  if (!trip) {
    return res.status(404).json({ error: 'Trip not found' });
  }
  return res.json(trip);
});

// Location Service: update driver location
app.post('/location/update', (req, res) => {
  const { driverId, lat, lng } = req.body || {};
  if (!driverId || typeof lat !== 'number' || typeof lng !== 'number') {
    return res.status(400).json({ error: 'driverId, lat, lng are required' });
  }
  const updated = state.geo.upsert({
    driverId,
    region: REGION,
    position: { lat, lng },
    updatedAt: new Date().toISOString(),
  });

  // Simulate publishing location event to Kafka
  // eslint-disable-next-line no-console
  console.log('[LocationEvent]', {
    region: REGION,
    driverId,
    lat,
    lng,
  });

  return res.status(200).json(updated);
});

// Dispatch Service: assign a driver to a trip
app.post('/dispatch/assign', (req, res) => {
  const { tripId } = req.body || {};
  if (!tripId) {
    return res.status(400).json({ error: 'tripId is required' });
  }

  const trip = state.trips.get(tripId);
  if (!trip) {
    return res.status(404).json({ error: 'Trip not found' });
  }

  const updated = assignDriver(state, trip);
  if (!updated || !updated.driverId) {
    return res.status(200).json({ message: 'No drivers available yet', trip: updated ?? trip });
  }

  return res.status(200).json(updated);
});

// Pricing Service: recompute price (e.g. if surge changes)
app.post('/pricing/reprice', (req, res) => {
  const { tripId } = req.body || {};
  if (!tripId) {
    return res.status(400).json({ error: 'tripId is required' });
  }
  const existing = state.trips.get(tripId);
  if (!existing) {
    return res.status(404).json({ error: 'Trip not found' });
  }
  const priceCents = calculatePriceCents(existing.origin, existing.destination);
  const updated = state.trips.update(tripId, { priceCents });
  return res.status(200).json(updated);
});

// Notifications Service: list recent notifications (for debugging)
app.get('/notifications', (_req, res) => {
  res.json(state.notifications.list());
});

const port = Number(process.env.PORT || (REGION === 'A' ? 4001 : 4002));
app.listen(port, () => {
  // eslint-disable-next-line no-console
  console.log(`Region service for region ${REGION} listening on port ${port}`);
});


