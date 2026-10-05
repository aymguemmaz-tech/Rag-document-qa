---
title: Incident Response Policy
owner: Engineering
---

# Incident Response Policy

This policy describes how Veloria Mobility detects, manages and learns from technical and safety incidents.

## Severity levels

| Severity | Definition | Acknowledge within | Status updates |
| --- | --- | --- | --- |
| SEV1 | City-wide outage: riders cannot unlock or ride, or there is a safety risk | 5 minutes | Every 30 minutes |
| SEV2 | Major degradation: payments failing for more than 10% of rides, or one zone unavailable | 15 minutes | Every 60 minutes |
| SEV3 | Minor degradation of a single feature | 4 business hours | Daily |
| SEV4 | Cosmetic issue with no rider impact | Next sprint | None |

Every SEV1 incident has a dedicated incident commander who coordinates the response and makes decisions.

## On-call

Engineering runs a weekly on-call rotation with a primary and a secondary engineer. The handover takes place every Monday at 10:00. If the primary engineer does not acknowledge a SEV1 page within 10 minutes, the page escalates to the secondary engineer and then to the Head of Engineering.

## Communication

For SEV1 and SEV2 incidents the public status page is updated by the incident commander or a delegate. For SEV1 incidents the city partner team in Arvenna must be informed within 1 hour.

## Safety incidents

Any incident in which a person is injured must be reported to the Safety Officer within 24 hours. If the injury is serious, the regulator is notified within 72 hours.

## Postmortems

A blameless postmortem is required for every SEV1 and SEV2 incident. It must be published within 5 business days of resolution. Action items from postmortems are tracked on the Reliability board and reviewed every two weeks.
