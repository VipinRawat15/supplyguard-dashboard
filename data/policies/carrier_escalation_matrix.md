# Carrier Escalation and Operational Capacity Matrix

## Purpose
Defines concrete escalation actions once a shipment has been confirmed
as genuinely at risk (post case-owner review), and how to allocate
limited daily review capacity across the ranked worklist.

## Escalation Ladder
1. **Monitor**: default state for Moderate and Low risk shipments. No
   action beyond automated tracking.
2. **Verify**: case owner confirms current carrier tracking status
   against the promised delivery window. Applies to all High and Very
   High risk shipments as the first response step.
3. **Expedite**: if verification confirms a real delay and the shipment
   is still in a fulfillment stage where mode/carrier can be changed,
   upgrade the shipping mode or switch carrier to the best-performing
   available option for that origin-destination lane.
4. **Communicate**: if expediting is not feasible (shipment already
   in transit, no faster option available), proceed directly to
   Customer Communication Policy notification.
5. **Escalate to Leadership**: if a specific carrier or lane shows a
   cluster of Very High risk shipments (e.g., more than 5 in a single
   day on the same carrier/lane combination), escalate to logistics
   leadership as a potential systemic carrier performance issue rather
   than continuing to handle each shipment individually.

## Daily Capacity Allocation Guidance
Operations teams rarely have capacity to manually review every flagged
shipment. When daily case-review capacity is limited, allocate it in
this order:
1. All Very High risk First Class and Same Day shipments (highest
   historical late-risk rate combined with least tolerance for delay).
2. Remaining Very High risk shipments, ranked by predicted probability.
3. High risk shipments, ranked by predicted probability, time permitting.

Analysis of the risk-ranked worklist shows that reviewing just the
top 20% of shipments by predicted risk typically captures roughly a
third of all actual late deliveries at well over 90% precision — so
even limited review capacity, correctly targeted using the model's
ranking, captures disproportionate operational value compared to
random or purely status-based triage.

## When Not to Escalate
Do not apply expedited-mode upgrades to shipments already flagged
CANCELED or SUSPECTED_FRAUD (see Fraud and Cancellation Policy), and do
not escalate Moderate or Low risk shipments outside the automatic
re-escalation triggers defined in the SLA Risk Response Policy — doing
so wastes limited operational capacity on shipments unlikely to need
it.
