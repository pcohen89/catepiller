# Caterpiller

## Different ideas for how to handle the data

### 1. Build data by taking summary statistics if multiple components of the same type match one assembly
For example, if an assembly has two nuts, in this data build we would aggregate all of the fields associated with those two nuts (like average their weights, take max of their weights, etc) 

Things still to do:
- [x] Encode categorical variables
- [x] Create field dictionary for component subtables
- [ ] Extract information from the names in type_connection, then merge on to adaptor

Summary statistics planned to use:
- [x] Average numeric columns and binary
- [x] Take max and min
- [x] Count the number of instances of that comp type matched to the assembly

### 2. Don't merge to the sub-component tables, just count instances of each component on the assembly
In this case, the data will have a feature for each component. The feature will count how many of that component are present on the assembly instance. Note, sometimes you have multiples of the exact same component noted in var component_quantity, so this approach will allow us to incorporate that information in a way that 1. misses

### 3. Try different outcome variables

- [x] Model with current data structure and include units as a field

### 4. Misc data ideas
- [x] Sum weight from all components

## Modeling approaches
- [x] Gradient deep trees (currently setting up a CV environment to tune one of these up)
- [x] Gradient with stumps
- [ ] Svm (small data means this might work okay)
- [ ] penalized regression
- [ ] NN (sigh)
- [ ] Try to get XGBOOST running

## Blending approaches
My current plan is to blend at the submission level. SO, I will try to create many submissions that are as good as possible using methods that are as different as possible and then do some sort of semi-naive blending of the final submissions. 


