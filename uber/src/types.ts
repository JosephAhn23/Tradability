export type RegionCode = 'A' | 'B';

export interface LatLng {
  lat: number;
  lng: number;
}

export type TripStatus = 'requested' | 'assigned' | 'ongoing' | 'completed' | 'cancelled';

export interface Trip {
  id: string;
  region: RegionCode;
  riderId: string;
  driverId?: string;
  origin: LatLng;
  destination: LatLng;
  etaMinutes: number;
  priceCents: number;
  currency: string;
  status: TripStatus;
  createdAt: string;
  updatedAt: string;
}

export interface DriverLocation {
  driverId: string;
  region: RegionCode;
  position: LatLng;
  updatedAt: string;
}


