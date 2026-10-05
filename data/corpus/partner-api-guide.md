---
title: City Partner API Guide (v2)
owner: Platform team
---

# City Partner API Guide (v2)

The City Partner API gives municipal partners real-time access to vehicle availability, parking zones and aggregated trip statistics. The base URL is `https://api.veloria.example/partner/v2`.

## Authentication

The API uses the OAuth 2.0 client credentials flow. Access tokens are valid for 60 minutes. Request a new token before the current one expires; refresh tokens are not issued.

## Rate limits

Each client may send 600 requests per minute, with bursts of up to 100 requests. Requests above the limit receive HTTP 429 with a Retry-After header.

## Endpoints

- `GET /vehicles` returns real-time vehicle availability. The vehicle feed is refreshed every 30 seconds.
- `GET /trips/aggregates` returns hourly trip aggregates. Every aggregate contains at least 10 trips.
- `GET /zones` returns parking and no-parking zones as GeoJSON.
- `POST /incidents` lets a city report a misparked or damaged vehicle.

After a city reports a misparked vehicle, Veloria removes it within 2 hours in the city centre and within 6 hours elsewhere.

## GBFS feed

A public feed in GBFS version 2.3 is available at `/gbfs/gbfs.json`. It does not require authentication.

## Pagination and errors

List endpoints use cursor-based pagination with a maximum page size of 500 items. Errors are returned as JSON problem details (RFC 9457).

## Versioning

Version 1 of the API is deprecated. Version 1 will be switched off on 31 December 2026. Breaking changes to version 2 are announced at least 6 months in advance on the partner mailing list.
