## Decision Tree Overview

### Primary Function:
- Used for classification and regression tasks in machine learning.
- Constructs a tree structure with internal nodes, branches, and leaf nodes.

### Tree Structure:
- Each internal node represents a test on a specific attribute.
- Branches depict the outcomes of these tests.
- Leaf nodes store class labels or values in regression cases.

### Tree Construction:
- Accomplished by recursively splitting the training data based on attribute values.
- The process stops upon meeting stopping criteria, such as maximum depth or minimum samples required to split a node.

### Attribute Selection:
- During training, the algorithm chooses the best attribute to split the data.
- Uses metrics like entropy or Gini impurity to measure the impurity level in subsets.
- Aims to find attributes that provide maximum information gain or reduce impurity after splitting.
