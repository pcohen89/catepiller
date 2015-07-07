# Caterpiller

## Different ideas for how to handle the data

### 1. Build data by taking summary statistics if multiple components of the same type match one assembly
For example, if an assembly has two nuts, in this data build we would aggregate all of the fields associated with those two nuts (like average their weights, take max of their weights, etc) 

Things still to do:
- [ ] Encode categorical variables
- [ ] Create field dictionary for component subtables

Summary statistics planned to use:
- [ ] Average numeric columns and binary
- [ ] Take max and min
- [ ] Count the number of instances of that comp type matched to the assembly

### 2. Don't merge to the sub-component tables, just count instances of each component on the assembly
In this case, the data will have a feature for each component. The feature will count how many of that component are present on the assembly instance. Note, sometimes you have multiples of the exact same component noted in var component_quantity, so this approach will allow us to incorporate that information in a way that 1. misses

### 3. Try different outcome variables

- [ ] Model with current data structure and include units as a field

### 4. Misc data ideas
- [ ] Sum weight from all components

## Modeling approaches

## Blending approaches


