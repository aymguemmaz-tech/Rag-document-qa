---
title: "Postmortem: Payment Authorisation Outage (14 April 2026)"
owner: Payments
severity: SEV2
---

# Postmortem: Payment Authorisation Outage (14 April 2026)

Incident commander: Lena Hartmann. Status: resolved. Severity: SEV2.

## Summary

On 14 April 2026, between 07:42 and 09:19 CEST, riders could not start rides because payment authorisation failed. The outage lasted 97 minutes during the morning commute.

## Impact

About 18% of unlock attempts failed during the incident window, a total of 6,214 failed unlock attempts. Roughly 2,900 riders were affected. Estimated lost revenue is €11,400. There was no safety impact.

## Timeline (CEST)

- 07:42 First failed payment authorisations.
- 07:51 Alert fired on the payment error rate, nine minutes after the first failures.
- 07:55 Incident declared as SEV2 and Lena Hartmann assigned as incident commander.
- 08:30 Investigation narrowed the problem to TLS handshake errors with the payment gateway.
- 09:12 New certificate deployed.
- 09:19 Error rate back to normal; recovery confirmed.

## Root cause

An intermediate TLS certificate on the connection between the ride service and our payment gateway provider, PayLane, had expired. The automatic certificate renewal job had been disabled during the February infrastructure migration and was never re-enabled.

## What went well

The payment error-rate alert detected the problem quickly, and the incident commander kept the status page current every 60 minutes.

## What went poorly

Nobody monitored certificate expiry dates, and the migration checklist did not include re-enabling scheduled jobs.

## Action items

1. Add certificate expiry monitoring with a 30-day warning. Owner: Platform team. Due: 15 May 2026.
2. Re-enable and test the certificate auto-renewal job. Owner: Infrastructure team. Due: 30 April 2026.
3. Allow riders in good standing to unlock with deferred payment when the gateway is down. Owner: Payments team. Due: Q3 2026.
