# Caterpiller

## Ideas for data build

### Build data by taking summary statistics if multiple components of the same type match one assembly
For example, if an assembly has two nuts, in this data build we would aggregate all of the fields associated with those two nuts (like average their weights, take max of their weights, etc) 

Things still to do:
- [ ] Encode categorical variables
- [ ] Create field dictionary for component subtables

Summary Statistics
- [ ] Average numeric columns and binary
- [ ] Take max and min
- [ ] Count the number of instances of that comp type matched to the assembly
