# Caterpiller (spelled correctly, unlike the repo)

## Different ideas for how to handle the data

### 1. Build data by taking summary statistics if multiple components of the same type match one assembly
For example, if an assembly has two nuts, in this data build we would aggregate all of the fields associated with those two nuts (like average their weights, take max of their weights, etc) 

Things still to do:
- [x] Encode categorical variables
- [x] Create field dictionary for component subtables
- [x] Extract information from the names in type_connection, then merge on to adaptor
- [x] Manual variable creation
- [x] Reshape the spec data to indicators

Summary statistics planned to use:
- [x] Average numeric columns and binary
- [x] Take max and min
- [x] Count the number of instances of that comp type matched to the assembly
- [x] Count instances of that component id in the whole data set and merge that on
- [x] Try data build that only takes one moment (min, median, max), first confirm that these moments are super highly correlated

### 2. Don't merge to the sub-component tables, just count instances of each component on the assembly
In this case, the data will have a feature for each component. The feature will count how many of that component are present on the assembly instance. Note, sometimes you have multiples of the exact same component noted in var component_quantity, so this approach will allow us to incorporate that information in a way that 1. misses

### 3. Try different outcome variables

- [x] Model with current data structure and include units as a field
- [x] Somehow account for the fact that some tube assmblies have 4-7 observations associated with them
- [x] Use 1/16th power rather than the 1 + log
- [x] Use 1/16th for stacking
- [x] Account for the fact the some tube assembly ids have multiple observations, which screws up weighting (how do you do the equiv of clustering for prediction)

### 4. Misc data ideas
- [x] Sum weight from all components
- [x] total weight/ tube length
- [x] sum all tube-ish lengths
- [x] wall thickness * length
- [x] bends/length
- [x] Flag if end_a != end_x
- [x] Number of bends per bend radius (does this even make sense?)
- [x] Unique or rare part should be interacted with quantity
- [x] Analyze bill of materials quantity more, somehow I need to capture that if a given component is adding a lot of cost, than the quantity of that component is really important (it did turn out to be very important)
- [x] maybe sum the number of tube assemblies associated with the supplier
- [ ] Identify if any tubes are exactly similar to any others (I have done this and it doesnt seem to help as much as expected, need to do further inquiry)
- [ ] Interact year with other major variables
- [x] 'other' seems to give good results, look into this component and manually extracting variables
- 

## Modeling approaches
- [x] Gradient deep trees (currently setting up a CV environment to tune one of these up)
- [x] Gradient with stumps (this seems to be dominated by meduim depth trees)
- [x] Svm (small data means this might work okay)
- [x] penalized regression
- [x] NN (sigh, finding a reasonable specification has been a nightmare)
- [x] KNN (this might actually be a reasonable application for it, remember to normalize first)(this was bad)
- [x] Try to get XGBOOST running
- [x] Two stage stacking approach (first stage you create predictions for bunch of model using only two components at a time, second state you fite a ridge to all varaibles plus first stage predictions)

## Blending approaches
My current plan is to blend at the submission level. SO, I will try to create many submissions that are as good as possible using methods that are as different as possible and then do some sort of semi-naive blending of the final submissions. 


