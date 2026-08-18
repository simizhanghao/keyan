# Day4 matched 5-seed audit

complete=True  missing=none  day5=unused

## File-Acc (%)

| Model           | seed0 | seed1 | seed2 | seed3 | seed4 | Mean±Std |
| --------------- | ----: | ----: | ----: | ----: | ----: | -------: |
| B Main          |  41.7 |  62.5 |   4.2 |   4.2 |   4.2 | 23.4±27.2 |
| C Full zscore   |  66.7 |  70.8 |  58.3 |  66.7 |  66.7 | 65.8±4.6 |
| C' Full ratio   |  70.8 |  87.5 |  70.8 |  79.2 |  87.5 | 79.2±8.4 |
| A CNN           |  75.0 |  75.0 |  58.3 |  62.5 |  58.3 | 65.8±8.6 |

## Paired Δ File-Acc (pp)

C−B   per seed: [25.0, 8.299999999999997, 54.099999999999994, 62.5, 62.5]   count C>B = 5/5
C'−B  per seed: [29.099999999999994, 25.0, 66.6, 75.0, 83.3]   count C'>B = 5/5
C−A   per seed: [-8.299999999999997, -4.200000000000003, 0.0, 4.200000000000003, 8.400000000000006]   count C>A = 2/5
C'−A  per seed: [-4.200000000000003, 12.5, 12.5, 16.700000000000003, 29.200000000000003]   count C'>A = 4/5

## Paired Δ Window-Acc (pp)

C−B   per seed: [5.699999999999999, -17.400000000000002, 25.5, 23.4, 32.5]
C'−B  per seed: [21.599999999999998, -4.900000000000006, 40.4, 41.9, 42.400000000000006]

Registered count gate is 4/5 and is not moved after seeing data.
verdict remains COMPUTED_COUNTS_ONLY until a human reads window+file together.
Experiment 2 / RCOF closed.
