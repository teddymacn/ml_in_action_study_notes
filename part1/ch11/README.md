# Apriori

Chapter 11 talks about Apriori algorithm, which is one of the most popular algorithms for association analysis. 

Apriori is a classical algorithm in data-mining. It is often used for mining frequent itemsets and relevant association rules from basket and transaction logs. 

It is very important and polular in traditional stores and e-commerce. It helps the customers in purchasing their items with more ease which increases the sales of the markets. It has also been used in the field of healthcare for the detection of adverse drug reactions (ADR), where it produces association rules that indicates what all combinations of medications and patient characteristics lead to ADRs.

The idea of Apriori is to identify characteristics of a data set and attempt to note how frequently the characteristics pop up throughout the set. The algorithm usually first defines a pre-arranged amount, called "minimal support", then a “frequent” data characteristic is one that occurs above that pre-arranged amount, known as a support. A confidence is defined for an association rule like {diapers} ➞ {wine}. For example, the confidence for the rule above is defined as support({diapers, wine})/support({diapers}), which is the percentage of this rule working in the dataset.

## Demo Code

[apriori.py](apriori.py) - Revised version of the original apriori demo