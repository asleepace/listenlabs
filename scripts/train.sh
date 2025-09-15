# !/bin/zsh

bun run neural train 30 150 \
  --assistGain=3 \
  --oracleRelabelFrac=1 \
  --elitePercentile=0.05 \
  --resume=true

bun run sanity

bun run neural resume 30 150 \
  --assistGain=2.5 \
  --oracleRelabelFrac=0.5 \
  --elitePercentile=0.05

bun run sanity

bun run neural resume 20 200 \
  --assistGain=1.5 \
  --oracleRelabelFrac=0.2 \
  --elitePercentile=0.10
