# FP-growth

In the end of previous chapter, we mentioned that the FP-growth algorithm is a significant improvement over Apriori for association analysis, which is the topic of this chapter.

Comparing to Apriori, FP-growth executes much faster, but unlike Apriori returning not only frequent itemsets but also association rules, FP-growth doesn't find association rules.

Why FP-growth is so fast is because it requires only two scans of the datasets whereas Apriori scans the dataset for every potential frequent item.

## Demo Code

[fpGrowth.py](fpGrowth.py) - Revised version of the original fpGrowth demo
