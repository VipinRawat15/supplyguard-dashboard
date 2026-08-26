# Fraud and Cancellation Handling Policy

## Purpose
Defines how to handle shipments where `Order Status` is CANCELED or
SUSPECTED_FRAUD, or where the SupplyGuard `Was_Canceled` flag is set.
These orders behave differently from normal in-progress shipments and
should not be routed through standard late-delivery escalation.

## Why These Are Handled Separately
Orders with these statuses never proceed to a normal shipping outcome —
historical data confirms 100% of CANCELED and SUSPECTED_FRAUD orders
have a Late_delivery_risk value of 0, because there is no shipment in
progress to be late. A low predicted late-delivery probability on one of
these orders does not mean "on track for on-time delivery" — it means
the model correctly recognizes the shipment isn't proceeding.

## SUSPECTED_FRAUD Handling
Route immediately to the fraud review team, bypassing the standard
logistics escalation queue entirely. Do not attempt shipping-mode
upgrades, carrier escalation, or customer delay notifications for
these orders — the priority is fraud verification, not delivery speed.
If fraud review clears the order, it re-enters the normal order pipeline
and should be re-scored by SupplyGuard from that point forward.

## CANCELED Handling
No delivery-related action is needed. Route to standard
cancellation/refund processing per finance team procedures. Confirm the
cancellation reason is logged for trend monitoring — a spike in
cancellations concentrated in a specific product category, region, or
shipping mode may indicate an upstream issue (e.g., inventory
mismatches or a pricing error) worth investigating separately from
delivery-risk operations.

## Dashboard Display Guidance
When displaying the SupplyGuard risk-ranked worklist, CANCELED and
SUSPECTED_FRAUD orders should be visually separated from the active
late-risk worklist (e.g., a distinct tab or filter) rather than mixed
into the main risk ranking, since their low predicted probability would
otherwise misleadingly appear alongside genuinely low-risk, actively
shipping orders.
