# Apriori

Chapter 11 talks about Apriori algorithm, Apriori is a classical algorithm in data-mining. It is often used for mining frequent itemsets and relevant association rules from basket and transaction logs. 

It is quite polular in traditional stores and e-commerce. It helps the customers in purchasing their items with more ease which increases the sales of the markets. It has also been used in the field of healthcare for the detection of adverse drug reactions (ADR), where it produces association rules that indicates what all combinations of medications and patient characteristics lead to ADRs.

The idea of Apriori is to identify characteristics of a data set and attempt to note how frequently the characteristics pop up throughout the set. This algorithm usually first defines a pre-arranged amount, called "minimal support", then a “frequent” data characteristic is one that occurs above that pre-arranged amount, known as a support. A confidence is defined for an association rule like {diapers} ➞ {wine}. For example, the confidence for the rule above is defined as support({diapers, wine})/support({diapers}), which is the percentage of this rule working in the dataset.

## Why Is It Not Being Used?

Well, although the Apriori algorithm is so simple and clear, it has some weaknesses. If the 1-itemset comes out to be very large, for ex. 10^4, then the 2-itemset candidate sets would be more than 10^7. Moreover, for a dataset with a large number of frequent items or with a low support value, the candidate itemsets will always be very large. These large datasets require a lot of memory to be stored in. Furthermore, Apriori algorithm also scans the database multiple times to calculate the frequency of the itemsets in k-itemset. So, Apriori algorithm turns out to be very slow and inefficient, especially when memory capacity is limited and the number of transactions is large.

Next chapter, we will talk about a significant improvement over Apriori for association analysis, [FP-growth algorithm](../ch12/README.md).

## Demo Code

[apriori.py](apriori.py) - Revised version of the original apriori demo