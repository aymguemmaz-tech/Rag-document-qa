---
title: Fleet Maintenance Manual
owner: Fleet Operations
---

# Fleet Maintenance Manual

The fleet consists of two vehicle models. The Veloria S3 e-scooter is limited to 25 km/h by law. The Veloria B2 e-bike provides pedal assistance up to 25 km/h.

## Maintenance levels

Maintenance is organised in three levels:

- **L1 (street)**: performed by field technicians on the street. Checks tyre pressure, brakes, lights, bell and the QR sticker.
- **L2 (workshop)**: performed at the workshop. Replaces brake pads, updates firmware and services bearings.
- **L3 (depot overhaul)**: performed at the depot. Inspects the frame for cracks and replaces motors when needed.

The workshop and the depot are both located at Depot North, Harbour Road 14.

## Service intervals

Every vehicle receives an L1 check every 7 days or every 150 km, whichever comes first. An L2 service is due every 1,000 km. An L3 overhaul is due every 6,000 km or every 12 months, whichever comes first.

## Tyres and brakes

Inflate e-scooter tyres to 50 psi and e-bike tyres to 45 psi. Brake pads are replaced when their thickness falls below 1.5 mm.

## Automatic grounding

The IoT module records a fall event when the accelerometer measures more than 4 g. After a fall event the vehicle is grounded automatically and cannot be rented until it passes an L1 inspection.

## Firmware updates

Firmware updates are rolled out in waves. A new version first goes to a 5% canary group for 48 hours, then to 25% of the fleet, and finally to the whole fleet. The rollout is rolled back if the vehicle crash rate increases by more than 0.5 percentage points.

## Spare parts

The workshop keeps a minimum stock of spare parts for two weeks of operation. Replacement motors have a supplier lead time of 21 days, so motor stock is reviewed every Monday.

## Performance targets

The fleet availability target is 92% of the fleet rideable between 06:00 and 22:00. The mean time to repair for L2 work should not exceed 36 hours.
