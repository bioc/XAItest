#' Add "simThresh" Simulated Threshold Feature
#' 
#' Adds a simulated feature to the dataset that has a p-value close to a specified threshold
#' when tested against the target variable. This function is useful for testing feature
#' selection methods and significance thresholds.
#' 
#' @param data A data frame containing the dataset.
#' @param target_column Character; name of the target column in the dataset. Default is "y".
#' @param target_type Character; specifies the type of the target variable. Can be "default", 
#'        "regression", or "classification". If "default", the function will automatically 
#'        determine the type based on the target column.
#' @param pval_target Numeric; the desired p-value for the simulated feature. Default is 0.049.
#' @param tolerance Numeric; the acceptable deviation from the target p-value. Default is 0.0005.
#' @param adj_bonf Logical; if TRUE, applies Bonferroni correction when calculating the p-value. Default is FALSE.
#' 
#' @return The original data frame with an additional column named "simThresh" containing the simulated feature.
#' 
#' @examples
#' # Example with a regression dataset
#' df <- data.frame(
#'   feature1 = rnorm(100),
#'   feature2 = rnorm(100, mean = 5),
#'   feature3 = runif(100, min = 0, max = 10),
#'   y = 1:100
#' )
#' 
#' # Add a simulated feature with p-value close to 0.05
#' df_with_sim <- addSimThresh(df, target_column = "y", pval_target = 0.05)
#'
#' # Check that the correlation between simThresh and the target variable is close to 0.05
#' print(cor.test(df_with_sim$simThresh, df_with_sim$y)$p.value)
#' 
#' # Example with a classification dataset
#' df_class <- data.frame(
#'   feature1 = rnorm(100),
#'   feature2 = rnorm(100, mean = 5),
#'   feature3 = runif(100, min = 0, max = 10),
#'   y = factor(c(rep("A", 50), rep("B", 50)))
#' )
#' 
#' # Add a simulated feature with p-value close to 0.01
#' df_class_with_sim <- addSimThresh(df_class, target_column = "y", 
#'                                  pval_target = 0.05)
#'
#' # Check that the p-value of the simThresh column is close to 0.05
#' print(t.test(simThresh ~ y, data = df_class_with_sim)$p.value)
#'
#' 
#' @export
addSimThresh <- function(data, target_column="y", target_type="default",
            pval_target=0.049, tolerance=0.0005, adj_bonf=FALSE) {
    if (target_type == "default") {
        # Check if target column exists in the data
        if (!(target_column %in% colnames(data))) {
            stop(paste("Target column", target_column, "not found in data"))
        }
        
        # Check if target column values are numeric or not
        if (!is.numeric(data[[target_column]])) {
            target_type <- "classification"
        } else {
            target_type <- "regression"
        }
    }
    y <- target_column
    if (target_type == "regression") {   
        # Use the target column name for further processing
        if (adj_bonf) {
        newSD <- .findNoiseSD(data[[y]], ncol(data)-1, sdSup=100,
                                pvalTarget=pval_target,
                                tolerance=tolerance)
        } else {
            newSD <- .findNoiseSD(data[[y]], 0, sdSup=100,
                                pvalTarget=pval_target,
                                tolerance=tolerance)
        }
        values <- data[[y]] + .regRNorm(length(data[[y]]), 0, newSD)
        dfSimu <- as.data.frame(values)
        colnames(dfSimu) = "simThresh"
        data[,"simThresh"] <- dfSimu
        #colnames(dfSimu) = "simFeat"
    }
    if (target_type == "classification") {
        categ1 <- unique(data[[y]])[1]
        categ2 <- unique(data[[y]])[2]
        n1 <- sum(data[[y]] == categ1)
        n2 <- sum(data[[y]] == categ2)
        nFeatures <- length(colnames(data)) - 1
        features <- colnames(data)[colnames(data) != y]
        mean <- mean(unlist(apply(t(data[,features]), 1, mean)))
        sd <- mean(unlist(apply(t(data[,features]), 1, sd)))
        if (adj_bonf) {
            gm <- .findGM(n1, n2, nFeatures, pvalTarget=pval_target, tolerance=tolerance)
        } else {
            gm <- .findGM(n1, n2, 0, pvalTarget=pval_target, tolerance=tolerance)
        }
        newDistrib <- .distribFromGM(gm, sd)
        vec1 <- .regRNorm(n1, mean - newDistrib$deltMean, newDistrib$newSD)
        vec2 <- .regRNorm(n2, mean + newDistrib$deltMean, newDistrib$newSD)
        dfSimu <- as.data.frame(c(vec1,vec2))
        colnames(dfSimu) = "simThresh"
        data[,"simThresh"] <- dfSimu
    }
    return(data)
}
