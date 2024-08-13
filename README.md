# Demo - MAB test with Thompson sampling

This demo uses the dataset from [Kaggle/ab-testing-dataset](https://www.kaggle.com/datasets/amirmotefaker/ab-testing-dataset/) to simulate the whole situation again and compare the result of the MAB and AB tests by limiting the daily sample size to 10k impressions. For the AB test, both variants always split the sample size equally (5k). Whenever the number of impressions in the data is less than the desired sample size, 5k in AB test or the calculated size in MAB, we assume that we can only collect the number of samples in the given data, with the concern that the CPI may increase as the number of impressions increases.

### Objective:
Optimise the number of purchases at the lowest cost

### Assumptions:
- The cost, Spend[USD], is calculated by the number of impressions.
- CPI (cost per impression) is always constant at the same time when fewer impressions have been obtained.
