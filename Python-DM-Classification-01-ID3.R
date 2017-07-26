############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Data Mining
# Lesson: Decision Trees - ID3

# Citation: 
# PEREIRA, V. (2018). Project: ID3, File: Python-DM-Classification-01-ID3.R, GitHub repository: <https://github.com/Valdecy/ID3>

############################################################################

# Code from: https://cran.r-project.org/web/packages/data.tree/vignettes/applications.html


################        Part 1 - Function      #############################

IsPure <- function(data) {
  length(unique(data[,ncol(data)])) == 1
}
Entropy <- function( vls ) {
  res <- vls/sum(vls) * log2(vls/sum(vls))
  res[vls == 0] <- 0
  -sum(res)
}
InformationGain <- function( tble ) {
  entropyBefore <- Entropy(colSums(tble))
  s <- rowSums(tble)
  entropyAfter <- sum (s / sum(s) * apply(tble, MARGIN = 1, FUN = Entropy ))
  informationGain <- entropyBefore - entropyAfter
  return (informationGain)
}
TrainID3 <- function(node, data) {
  node$obsCount <- nrow(data)
  
  # if the data-set is pure then
  if (IsPure(data)) {
    
    # construct a leaf having the name of the pure feature
    child <- node$AddChild(unique(data[,ncol(data)]))
    node$feature <- tail(names(data), 1)
    child$obsCount <- nrow(data)
    child$feature <- ''
  } else {
    
    # calculate the information gain
    ig <- sapply(colnames(data)[-ncol(data)], function(x) InformationGain(table(data[,x], data[,ncol(data)]))
    )
    # chose the feature with the highest information gain
    # if more than one feature have the same information gain, then take
    # the first one
    feature <- names(which.max(ig))
    node$feature <- feature
    
    # take the subset of the data-set having that feature value
    childObs <- split(data[ ,names(data) != feature, drop = FALSE], data[ ,feature], drop = TRUE)
    
    for(i in 1:length(childObs)) {
      
      # construct a child having the name of that feature value
      child <- node$AddChild(names(childObs)[i])
      
      # call the algorithm recursively on the child and the subset      
      TrainID3(child, childObs[[i]])
    }
  }
}
Predict <- function(tree, features) {
  if (tree$children[[1]]$isLeaf) return (tree$children[[1]]$name)
  child <- tree$children[[features[[tree$feature]]]]
  return ( Predict(child, features))
}

############### End of Function ##############

######################## Part 2 - Usage ####################################

# For the following implementation, the attributes are in columns 1 to n-1,
# whereas the target is in the last column.

my_data <- as.data.frame(Python_DM_Classification_01_ID3)

tree <- Node$new("Dataset")
TrainID3(tree, my_data)
print(tree, "feature", "obsCount")

Predict(tree, c(Outlook = 'Rainy', Temperature = 'Hot', Humidity = 'High', Windy = 'True'))

