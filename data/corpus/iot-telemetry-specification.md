---
title: IoT Telemetry Specification (VTM-4)
owner: Hardware team
---

# IoT Telemetry Specification (VTM-4)

Every Veloria vehicle carries a VTM-4 telematics module. This specification describes what the module measures and how it communicates with the platform.

## Hardware

The VTM-4 combines an LTE-M modem, a multi-constellation GNSS receiver, a 3-axis accelerometer and a Bluetooth Low Energy radio used for unlocking when the network is weak. A backup battery keeps the module alive for 72 hours when the main pack is removed.

## Sensors

The accelerometer samples at 100 Hz. Fall detection runs on the module itself, and the fall threshold is configured by Fleet Operations. Geofences are evaluated on the module once per second, so speed limits apply even without network coverage.

## Messaging

Telemetry is sent with MQTT over TLS 1.3 to the Veloria message broker. A typical telemetry message is 180 bytes. Parked vehicles send a lightweight heartbeat every 5 minutes so the platform knows the module is online. Location reporting frequency is governed by the Data Privacy and Retention Policy.

## Commands

The platform can lock, unlock, sound the alarm and set the speed limit remotely. Every command is signed and acknowledged by the module within 3 seconds; unacknowledged commands are retried three times.

## Firmware

Module firmware is updated over the air using signed images and an A/B partition scheme, so a failed update boots back into the previous version automatically.
