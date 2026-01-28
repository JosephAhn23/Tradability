## Uber-like Multi-Region Backend (Prototype)

This project is a **code prototype** of the multi-region Uber-like architecture you described:

- **API Gateway** that chooses a region (A/B) for each request.
- **Per-region stateless services** (in a single process per region for now) for:
  - Auth, Trip, Dispatch, Location, ETA, Pricing, Payments, Notifications, Ratings, Fraud.
- Simple **in-memory data layer** to keep the demo lightweight but structured so you can later swap in Postgres, Redis, and Kafka.

This is not production-ready, but it shows **how to turn the architecture into running software** with clear boundaries and flows.

### Tech stack

- **Language**: TypeScript (Node.js)
- **Framework**: Express
- **Dev tooling**: ts-node-dev, concurrently

### Services in this prototype

- `API Gateway`
  - Accepts client-style calls.
  - Chooses region based on simple `region` query parameter or header.
  - Proxies requests to region services.
- `Region Service` (one instance per region)
  - Runs as `REGION=A` and `REGION=B`.
  - Exposes routes that represent the regional microservices:
    - `/trip/*`
    - `/location/*`
    - `/dispatch/*`
    - `/pricing/*`
    - `/notifications/*`
    - (Auth, Ratings, Fraud are stubbed for now.)

### Core flows implemented

- **Trip request**
  - `POST /api/trip/request?region=A`
  - Creates a trip, calls ETA and Pricing submodules, stores trip in an in-memory store.
- **Location update**
  - `POST /api/location/update?region=A`
  - Updates driver location, writes to in-memory "Redis GEO" abstraction, and emits a fake event.
- **Dispatch lookup**
  - `POST /api/dispatch/assign?region=A`
  - Looks up nearby drivers from the in-memory geo store and assigns the closest.
- **Notifications**
  - Writes to an in-memory "notification outbox" and logs to console (simulating integration with a messaging provider).

### Running locally

1. Install dependencies:

```bash
npm install
```

2. Run all services (gateway + region A + region B):

```bash
npm run dev
```

3. Example requests (use `region=A` or `region=B`):

- **Request a trip**:

```bash
curl -X POST "http://localhost:3000/api/trip/request?region=A" ^
  -H "Content-Type: application/json" ^
  -d "{ \"riderId\": \"r1\", \"origin\": {\"lat\": 37.77, \"lng\": -122.42}, \"destination\": {\"lat\": 37.78, \"lng\": -122.41} }"
```

- **Update driver location**:

```bash
curl -X POST "http://localhost:3000/api/location/update?region=A" ^
  -H "Content-Type: application/json" ^
  -d "{ \"driverId\": \"d1\", \"lat\": 37.77, \"lng\": -122.42 }"
```

- **Assign driver (dispatch)**:

```bash
curl -X POST "http://localhost:3000/api/dispatch/assign?region=A" ^
  -H "Content-Type: application/json" ^
  -d "{ \"tripId\": \"<TRIP_ID_FROM_PREVIOUS_CALL>\" }"
```

### Next steps / how to grow this

- Swap the in-memory stores for:
  - **Postgres** (trip, user, pricing, ratings tables).
  - **Redis GEO** (driver locations and surge).
  - **Kafka** (events between services and regions).
- Split the region service into separately deployable microservices (trip, dispatch, location, etc.) behind a regional load balancer.
- Add authentication/authorization and rate-limiting at the API Gateway.


