# Identification guide

The package is organized around the view that boards hire from a feasible set rather than a frictionless national market. The main source of variation comes from exogenous local expansions in access to CEO-ready candidates due to clean release events.

## Core identification logic

1. Source firms experience clean standalone-role destruction, for example via merger or acquisition.
2. That event releases CEO-ready external executives into a local market.
3. Focal succession firms in the same feasible market face a larger accessible set even if their own fundamentals do not change.
4. If boards are constrained by search frictions, successor origin, search breadth, realized fit, and post-succession outcomes should move with that access shock.

The current package implements the paper around firms that already experience a CEO succession. It is intentionally a successor-choice design, not a turnover-incidence design.

## What the package treats as exogenous

The package does not assume every delisting or executive departure is an exogenous access shock. The main sample is built from tier-A releases:

- merger/acquisition induced loss of standalone status
- reliable timing and location
- plausible role elimination

Tier-B and tier-C events can be carried for robustness or diagnostics, but the baseline supply measures emphasize tier A.

## Assumptions baked into construction

### Pre-event information only

All explanatory variables are built with information available by `t-1`:

- internal bench from the focal top team at `t-1`
- released-candidate eligibility from source-firm roles at `t-1`
- portable candidate history from observed public-firm records through the candidate's availability year
- firm needs from focal-firm `t-1` outcomes and controls

### No silent fuzzy merging

Person resolution uses:

- normalized names
- reviewed crosswalks when supplied
- exact-name AUTO ids as a conservative fallback

The package never silently merges executives using fuzzy string matching alone.

### Geography shapes access

The default feasible set uses the specification layer's primary market definition:

- 60-mile radius around the focal HQ for the main released pool
- 100-mile, MSA, and same-state variants for robustness

This is deliberately central to the design; geography is not a peripheral robustness check.

### Out-of-industry main pool

The main released-candidate pool excludes same-FF10 source firms:

\[
R_{e,730d,60mi,outind} =
\{ i : 0 < a_e - r_i \le 730,\ \text{distance}_{if} \le 60,\ source\_ff10_i \neq focal\_ff10_f \}
\]

This keeps the baseline access shock from mechanically proxying for very local same-industry product-market events.

## Optional mechanism modules

The following are enrichments, not identification requirements:

- BoardEx-based `known_to_board` ties
- predictive fit
- shift-share instruments
- travel-time shocks
- BLM bridge variables
