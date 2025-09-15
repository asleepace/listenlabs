<img src="https://preview.redd.it/found-this-puzzle-on-a-billboard-in-sf-i-tried-feeding-it-v0-qe2xwiks2smf1.jpeg?width=1080&crop=smart&auto=webp&s=ea76455577cb6d9704b3761d9a158f4b7663bc75" />

# Berghain Challenge

The challenge has ended and I placed **96th** overall on the [leaderboard](https://berghain.challenges.listenlabs.ai/).

| Place                                        | Name                                    | Scenario 1 | Scenario 2 | Scenario 3 | Total |
| -------------------------------------------- | --------------------------------------- | ---------- | ---------- | ---------- | ----- |
| 98 │ Colin Teahan │ 805 │ 3459 │ 4340 │ 8604 |
| 400                                          | gpt-5-pro │ 1140 │ 7017 │ 16756 │ 24913 |

```bash
# download the repo & install deps (mainly just axios)
git clone https://github.com/asleepace/listenlabs.git
bun i

# run one of the specific scenarios
bun run game --scenario 2

# various neural-net commands (training)
bun run game neural validate
bun run game neural train
bun run game nueral resume
bun run game neural curriculum
```

**NOTE**: Currently the neural net bouncer is only trained on Scenario 2 and will not work for the others, if you want to try a different scenario please swap out the bouncer in `src/index.ts` in the `.withBouncer(() => { /* ... */})` block.

## Table of Contents

This repo contains several different algorithms for the "bouncers" used in this challenge, ranging from base logic to neaural nets.

- [bouncers](./src/bouncers/): deprecated bouncers from previous versions
- [cli](./src/cli/): command line interface helpers for the main app
- [conf](./src/conf/): shared configuration files
- [example](./src/example/): assorted code snippets
- [math](./src/math/): math utilities for common operations
- [neural-net](./src/neural-net/): everything neural net related (latest bouncer)
- [utils](./src/utils/): utilities for reading stuff from disk, logging, etc.

The following are important files:

- [berghain.ts](./src/berghain.ts): used to create games, make api calls, etc.
- [index.ts](./src/index.ts): main entry point where the app is run

Along with these important data directories:

- [data/](./data/): generic data produced from different places in the app
- [data/samples/](./data/samples/): real challenge data recorded while playing
- [data/best/](/data/best/): best weights from training neural net.
- [bouncer-data/](/bouncer-data/): training data for neural net

## Neural Net

The latest and best version of the algorithm used in this challenge was the neural net implementation, which is found
in the neural net directory.

Please see the [`INFO.md`](./src/neural-net/INFO.md) in that folder for more info.

## Related

- https://www.reddit.com/r/OpenAI/comments/1n6o5or/found_this_puzzle_on_a_billboard_in_sf_i_tried/#lightbox

## Bandit Configuration (Deprecated)

### HIGH IMPACT (tune these first):

1. MIN_THRESHOLD (Range: 0.2 - 0.7)

Impact: Controls base admission strictness
Sweet spot: 0.3 - 0.5
Lower = more lenient early (admit more people)
Higher = stricter (risk not filling venue)

2. TARGET_RANGE (Range: 2000 - 6000)

Impact: When you aim to complete quotas
Sweet spot: 3500 - 4500
Lower = rush to fill quotas early (risk running out of spots)
Higher = spread out (risk missing rare attributes)
Depends on rarest attribute frequency

3. URGENCY_MODIFIER (Range: 1.0 - 6.0)

Impact: How much being behind schedule matters
Sweet spot: 2.5 - 4.0
Lower = relaxed about timing (risk missing quotas)
Higher = panic when behind (risk over-admitting)

### MEDIUM IMPACT (fine-tune after high impact)

4. THRESHOLD_RAMP (Range: 0.2 - 0.8)

Impact: How threshold changes as venue fills
Sweet spot: 0.4 - 0.6
Lower = consistent threshold throughout
Higher = gets much stricter as you fill up

5. MULTI_ATTRIBUTE_BONUS (Range: 0.3 - 1.5)

Impact: Reward for people with multiple needed attributes
Sweet spot: 0.5 - 0.8
Too high = over-value "jack of all trades"
Too low = miss efficient multi-quota fills

6. CRITICAL_THRESHOLD (Range: 10 - 100)

Impact: When to panic about unfilled quotas
Sweet spot: 30 - 60
Lower = panic mode engages late
Higher = conservative/safe approach

### LOW IMPACT (minor tweaks)

7. CORRELATION_BONUS (Range: 0.1 - 0.5)

Impact: Reward for positively correlated attributes
Sweet spot: 0.2 - 0.3
Minor effect on overall strategy

8. NEGATIVE_CORRELATION_BONUS (Range: 0.3 - 1.0)

Impact: Reward for rare combinations
Sweet spot: 0.5 - 0.7
Helps with edge cases

9. RARE_PERSON_BONUS (Range: 0.3 - 1.0)

Impact: Extra boost for rare combos
Similar to NEGATIVE_CORRELATION_BONUS

10. NEGATIVE_CORRELATION_THRESHOLD (Range: -0.7 to -0.3)

Impact: What counts as "negatively correlated"
Sweet spot: -0.4 to -0.5
Very minor impact

## How to run?

```bash
bun install

bun run index.ts
```

## About

This project was created using `bun init` in bun v1.2.20. [Bun](https://bun.com) is a fast all-in-one JavaScript runtime.
