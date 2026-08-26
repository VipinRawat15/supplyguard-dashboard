# Geographic and Cross-Border Shipping Policy

## Purpose
Defines handling considerations for at-risk shipments based on market
and region, where logistics infrastructure and customs processes differ
from the domestic baseline.

## Market-Level Considerations
While market-level late-risk rates are relatively close to the overall
average in aggregate (all markets historically fall within roughly
54-56% late-risk rate), individual regions within a market can still
carry distinct operational constraints that matter once a shipment is
already flagged as high-risk by the model:

- **LATAM and Africa markets**: customs clearance delays are a common
  root cause of shipping delays that are not visible in the SLA-window
  data alone. When a High or Very High risk shipment in these markets is
  reviewed, case owners should check customs documentation completeness
  as a first step before assuming a carrier-side delay.
- **Europe and Pacific Asia markets**: delays are more commonly
  carrier-capacity related during regional peak shopping periods.
  Case owners should check for known regional carrier capacity alerts
  before escalating to a carrier switch.
- **USCA (US/Canada) market**: typically the most predictable market;
  an unexpected High or Very High risk score here is more likely
  attributable to shipping-mode or order-status factors than to
  regional logistics issues, and should be investigated accordingly
  using the shipment's top risk drivers rather than assumed to be a
  geography problem.

## Order Region Escalation Note
Order Region has a modest but non-trivial effect on predicted risk once
combined with other factors, even though no single region stands out
dramatically on its own. Case owners should treat region as a secondary
diagnostic signal — useful for explaining *why* a shipment is flagged,
less useful as a standalone trigger for escalation on its own.

## Cross-Border Documentation Checklist
For any Very High risk shipment crossing an international border,
confirm the following before considering the shipment a true logistics
delay requiring carrier escalation:
1. Customs declaration is complete and matches the shipped product
   category.
2. Product category is not subject to known regional import
   restrictions or additional inspection requirements.
3. Destination address includes a complete, correctly formatted postal
   code — incomplete address data is a common, easily fixed cause of
   customs hold delays.
