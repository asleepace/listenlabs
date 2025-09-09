# Adaptive Bouncer Algorithm

A dynamic admission control system for quota-based capacity management with multi-attribute constraints and correlation handling.

## Overview

The Bouncer Algorithm solves the problem of admitting people to a venue while satisfying multiple attribute quotas (e.g., 650 techno_lovers, 300 creative types, 750 berlin_locals) from a stream of candidates with unknown attribute combinations and correlations.

**Core Challenge**: Attributes may be correlated (positively or negatively), have different rarities, and require different quantities, making naive admission strategies fail to complete quotas.

## Architecture

### Components

1. **Metrics Engine** - Tracks quota progress, attribute frequencies, correlations, and risk assessment
2. **Score Calculator** - Evaluates candidate value based on urgency, rarity, and critical status
3. **Deflation Controller** - Dynamically adjusts admission rate to hit target efficiency
4. **Critical Detection** - Identifies attributes at risk of non-completion
5. **Endgame Handler** - Specialized logic for final quota completion

### Algorithm Flow

```
1. Calculate candidate score based on useful attributes
2. Apply deflation factor based on current admission rate
3. Check critical attribute requirements
4. Compare final score against dynamic threshold
5. Apply endgame logic if approaching capacity limits
```

## Key Algorithms

### Dynamic Scoring

Candidates receive scores based on:

- **Urgency**: `needed / urgencyDivisor`
- **Rarity**: Higher bonuses for low-frequency attributes
- **Progress**: Boost for lagging quotas
- **Critical Multiplier**: 5-15x boost for at-risk attributes
- **Multi-attribute Bonus**: Efficiency bonus for multiple useful attributes

### Deflation Control

Admission rates are controlled via score deflation:

```typescript
ratio = currentRate / targetRate
if (ratio > 1.3) deflationFactor = 0.65 // Over-admitting
if (ratio < 0.8) deflationFactor = 1.25 // Under-admitting
```

### Overfill Protection

Attributes are excluded from scoring when they approach completion:

```typescript
overfillThreshold = Math.max(0.88, Math.min(0.96, 0.82 + frequency * 0.3))
```

This gives rare attributes (6% frequency) an 88% threshold vs common attributes (60% frequency) a 96% threshold.

### Critical Detection

Attributes become critical when:

- Need > 15% of remaining capacity (capacity critical)
- Need > 90% of estimated remaining candidates with attribute (scarcity critical)
- Risk assessment flags them based on frequency vs requirements

### Endgame Logic

When `totalNeeded >= spotsLeft`, switches to quota-specific scoring that prioritizes exact needs over general desirability.

## Configuration

### Key Parameters

- `BASE_THRESHOLD`: Starting admission threshold (0.42-0.45)
- `TARGET_RATE`: Target admission percentage (25% = 1000/4000)
- `TARGET_RANGE`: Total candidates to process (4000)
- `MAX_CAPACITY`: Venue capacity (1000)

### Scoring Presets

- **CONSERVATIVE**: Lower bonuses, higher thresholds for selective admission
- **AGGRESSIVE**: Higher multipliers for faster quota completion
- **BALANCED**: Default middle-ground approach

## Edge Cases & Limitations

### Mathematical Impossibility

When sum of minimum quotas exceeds capacity (e.g., need 1800 people for 1000 spots), algorithm assumes attribute overlap and continues. Monitor `totalNeeded >= spotsLeft` for early warning.

### Negative Correlations

Strong negative correlations (e.g., -0.65 between techno_lover and berlin_local) can create impossible combinations. Algorithm uses critical detection and endgame logic to handle this.

### Rare Attribute Bottlenecks

Attributes with <10% frequency can create endgame bottlenecks. Dynamic overfill protection and critical multipliers address this, but may require manual threshold adjustment.

### Score Inflation

High critical multipliers (15x) can cause score inflation. Deflation controller and logarithmic normalization prevent runaway scores.

### Early Endgame Triggering

Overly aggressive endgame conditions can trigger too early. Current condition `spotsLeft <= 20 && totalNeeded <= spotsLeft * 2` prevents this.

## Performance Targets

- **Admission Rate**: 20-25% (configurable via TARGET_RATE)
- **Quota Completion**: >99% of all attributes
- **Efficiency**: <5% waste from overfilled quotas
- **Rejection Count**: ~3000 for 4000 candidates processed

## Monitoring

Key metrics to track:

- `admissionRate` vs `targetRate`
- `quotas[].needed` for incomplete attributes
- `deflationFactor` for rate control effectiveness
- `criticalAttributes.length` for bottleneck detection
- `rejectionGap` for efficiency measurement

## Common Issues

### Over-admission (>30% rate)

- Increase BASE_THRESHOLD
- Use CONSERVATIVE scoring preset
- Check deflation factor is applying correctly

### Under-admission (<20% rate)

- Decrease BASE_THRESHOLD
- Use AGGRESSIVE scoring preset
- Verify critical detection isn't too restrictive

### Incomplete Quotas

- Check attribute frequencies match actual distribution
- Verify overfill thresholds aren't excluding needed attributes too early
- Monitor for early endgame triggering

### Bottlenecks

- Increase critical multipliers for rare attributes
- Lower overfill threshold for problematic attribute
- Check for correlation conflicts requiring manual adjustment

## Algorithm Complexity

- **Time**: O(1) per admission decision
- **Space**: O(n) where n = number of attributes
- **Convergence**: Typically stabilizes within 500-1000 decisions
