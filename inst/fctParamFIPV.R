    model_type.rpart <- function(x, ...) {
        if (x$method == "class")  "classification" else  "regression"
    }
    predict_model.rpart <- function(x, newdata, type = NULL, ...) {
    if (model_type(x) == "classification") {
        probs <- predict(x, newdata, type = "prob", ...)
        return(as.data.frame(probs))
    }
    vals <- predict(x, newdata, type = "vector", ...)
    return(data.frame(Response = vals))
    }
    model_type.keras.engine.training.Model <- function(x, ...) {
        if (length(x$output_shape) >= 2 && x$output_shape[[2]] > 1) {
            "classification"
        } else {
            "regression"
        }
    }
    predict_model.keras.engine.training.Model <- function(x, newdata, type = NULL, ...) {
        m <- as.matrix(newdata)
        preds <- predict(x, m)
        if (model_type(x) == "classification") {
            as.data.frame(preds)
        } else {
            data.frame(Response = preds[,1])
        }
    }
    model_type.keras.engine.sequential.Sequential <- function(x, ...) {
        if (length(x$output_shape) >= 2 && x$output_shape[[2]] > 1) "classification" 
        else                                            "regression"
    }
    predict_model.keras.engine.sequential.Sequential <- function(x, newdata, type = NULL, ...) {
    m <- as.matrix(newdata)
    preds <- predict(x, m)
    if (model_type(x) == "classification") as.data.frame(preds) 
        else                              data.frame(Response = preds[,1])
    }
    model_type.keras.engine.training.Model <- model_type.keras.engine.sequential.Sequential
    predict_model.keras.engine.training.Model <- predict_model.keras.engine.sequential.Sequential
    library(kernlab)
    model_type.ksvm <- function(x, ...) {
        "classification"
    }
    predict_model.ksvm <- function(x, newdata, type = NULL, ...) {
        probs <- kernlab::predict(x, newdata, type = "probabilities", ...)
        df <- as.data.frame(probs)
        colnames(df) <- x@lev
        df
    }

library(lime)
library(randomForest)

# 1) Dire à lime si c'est classification ou regression
model_type.randomForest <- function(x, ...) {
  if (x$type %in% c("classification", "regression")) {
    x$type
  } else {
    stop("Unsupported randomForest type: ", x$type)
  }
}
# on alias aussi pour la classe formula
model_type.randomForest.formula <- model_type.randomForest


# 2) Expliquer à lime comment prédire
predict_model.randomForest <- function(x, newdata, type = NULL, ...) {
  # pour classification : renvoyer un data.frame de probabilités
  if (x$type == "classification") {
    probs <- predict(x, newdata = newdata, type = "prob", ...)
    return(as.data.frame(probs))
  }
  # pour regression : une colonne "Response"
  vals <- predict(x, newdata = newdata, type = "response", ...)
  data.frame(Response = vals)
}
predict_model.randomForest.formula <- predict_model.randomForest


featureImportanceLime <- function(data, y="y", featImpAgr = "mean",
                                    modelType = "classification",
                                    caretMethod="rf",
                                    caretTrainArgs=NULL,
                                    ml_algo="rf") {
    categ = y

    matX <- as.data.frame(data[, colnames(data) != y])
    vecY <- data[[y]]

    if ( ml_algo == "rf"){
        #carArgs1 <- list(x=matX, y=vecY, method=caretMethod)
        #model <- do.call(caret::train, c(carArgs1, caretTrainArgs))

        model <- randomForest::randomForest(as.formula(paste(categ, "~ .")), data = data, importance = TRUE)

        
    }

    if(ml_algo == "dt"){
        my_method <- "class"
        if (modelType == "regression"){
            my_method <- "anova"
        }
        model <- rpart::rpart(as.formula(paste(y, "~ .")), data = data, method = my_method)
    }

    if (ml_algo == "mlp"){
        library(keras)
        temp_df <- data
        
        cols_to_scale <- setdiff(names(temp_df), categ)
        temp_df[cols_to_scale] <- scale(temp_df[cols_to_scale])

        col_means <- colMeans(temp_df[cols_to_scale])
        col_sds <- apply(temp_df[cols_to_scale], 2, sd)

        n <- nrow(temp_df)
        train_data <- temp_df
        test_data  <- temp_df

        # Préparation des données :
        # Séparer les variables explicatives et la variable cible
        X_train <- as.matrix(train_data[, setdiff(names(train_data), categ)])
        X_test  <- as.matrix(test_data[, setdiff(names(test_data), categ)])

        # Pour le modèle Keras, il faut encoder la variable cible en one-hot
        if (modelType == "classification"){
            num_classes <- length(levels(temp_df[[categ]]))
            # Convertir les classes en indices (commençant par 0)
            y_train <- as.integer(train_data[[categ]]) - 1  
            y_test  <- as.integer(test_data[[categ]]) - 1

            # Encodage one-hot
            y_train <- to_categorical(y_train, num_classes = num_classes)
            y_test  <- to_categorical(y_test, num_classes = num_classes)
        }
        if (modelType == "regression"){
            y_train <- train_data[[categ]]
            y_test  <- test_data[[categ]]
        }

        # Construction du modèle MLP
        if (modelType == "classification"){
            model <- keras_model_sequential() %>%
            # Première couche cachée avec 4 neurones et activation ReLU
            layer_dense(units = 16, activation = "relu", input_shape = ncol(X_train)) %>%
            # Dropout classique (20% des neurones désactivés aléatoirement)
            layer_dropout(rate = 0.2) %>%
            # Deuxième couche cachée avec 2 neurones et activation ReLU
            layer_dense(units = 8, activation = "relu") %>%
                layer_dropout(rate = 0.2) %>%
                
            # Deuxième couche cachée avec 2 neurones et activation ReLU

            # Couche de sortie : softmax pour la classification multi-classes
            layer_dense(units = num_classes, activation = "softmax")
        } else {
            model <- keras_model_sequential() %>%
            layer_dense(
                units = 32, 
                activation = "relu", 
                input_shape = ncol(X_train),
                kernel_regularizer = regularizer_l2(l = 0.001)  # Régularisation L2
            ) %>%
            layer_dropout(rate = 0.2) %>%
            
            layer_dense(
                units = 16, 
                activation = "relu",
                kernel_regularizer = regularizer_l2(l = 0.001)  # Régularisation L2
            ) %>%
            
            # Couche de sortie pour la régression (1 neurone, activation linéaire par défaut)
            layer_dense(units = 1)
        }

        if (modelType == "classification"){
            my_loss <- "categorical_crossentropy"
            model %>% compile(
                loss = my_loss,
                optimizer = optimizer_adam(),
                metrics = c("accuracy")
            )
        }
        if (modelType == "regression"){
            my_loss <- "mean_squared_error"
            model %>% compile(
                loss = my_loss,
                optimizer = optimizer_adam(learning_rate = 0.01),
                metrics = c("mean_absolute_error")
            )
        }

        history <- model %>% fit(
            X_train, y_train,
            epochs = 5000,        # ajustez le nombre d'epochs selon vos besoins
            #epochs = 100, 
            batch_size = 32,       # ajustez la taille de batch
            validation_split = 0.2,
            verbose = 0
        )
        pred_wrapper <- function(object, newdata, ...) {
            predict(object, newdata)[, 1]
        }
        matX <- as.data.frame(X_train)
    }
    if (ml_algo == "svm-vanilladot" || ml_algo == "svm-rbfdot"){
        str_kern <- unlist(strsplit(ml_algo, 'svm-'))[2]
        #str_kern <- "rbfdot"
        model <- kernlab::ksvm(as.formula(paste(categ, "~ .")), 
                       data = data, 
                       kernel = str_kern,   # Noyau polynomial rbfdot polydot
                       prob.model = TRUE)
        X <- data[, setdiff(colnames(data), categ)]
        pred_fun <- function(model, newdata) {
            probs <- kernlab::predict(model, newdata, type = "probabilities")
            return(probs[, 1])
        }
    }
    
    explainer <- lime::lime(matX, model)
    
    if (modelType == "classification") {
        explanation <- lime::explain(matX, explainer,
                                        n_labels = 1, n_features = ncol(matX))
    } else {
        explanation <- lime::explain(matX, explainer, n_features = ncol(matX))
    }
    
    results <- unlist(lapply(colnames(matX), function(x) {
        myMask <- explanation$feature == x
        if (sum(myMask) == 0) {
            return(0)
        } else {
            featWeights <- explanation[myMask, "feature_weight"]
            if (featImpAgr == "mean") {
                return(mean(unlist(abs(featWeights))))
            }
            if (featImpAgr == "max_abs") {
                return(max(unlist(abs(featWeights))))
            }
        }
    }))
    
    names(results) <- colnames(data)[colnames(data) != y]
    return(results)
    #list(featImps = results, model = model)
}


# Assuming df1 is your data frame and 'categ' is the target column name (as a string)
# Example (if you need to simulate data, uncomment the next lines):
# set.seed(123)
# n <- 200   # number of observations
# p <- 5     # number of predictor variables
# X <- data.frame(matrix(rnorm(n * p), nrow = n))
# names(X) <- paste0("X", 1:p)
# df1 <- data.frame(y = with(X, 3 * X1 - 2 * X2 + rnorm(n)), X)
# categ <- "y"

# 1. Fit Random Forest model on the original data using the target variable 'categ'
fctParamFIPV <- function(data, categ="y", modelType="default", ml_algo="rf", algo_options=list(),
                fi_algo="accuracy", fi_transfo="none",pv_algo="pimp", adjMethod="bonferroni",
                n_perm=100, n_probes=10, shap_nsim=100, verbose=0){

    if ( pv_algo == "PIMP"){
        pv_algo <- "pimp"
    }
    if ( pv_algo == "mProbes" || pv_algo == "mprobes" ){
        pv_algo <- "probes"
    }
    
    if (!modelType %in% c("classification", "regression", "default")){
        stop("The modelType must be a 'classification',
                    'regression' or 'default'")
    }
    if (modelType == "default"){
        if (is(data[[categ]], "character")){
            modelType <- "classification"
        }
        if (is.numeric(data[[categ]])){
            modelType <- "regression"
        }
    }
    computeTimes <- list()
    predictor_names <- setdiff(names(data), categ)
    data[,setdiff(colnames(data), "y")] <- scale(data[,setdiff(colnames(data), "y")])
    data <- data[sample(1:nrow(data)),]

    jitter_ratio <- 0.00001

    if (modelType == "classification"){
        data[[categ]] <- as.factor(data[[categ]])
    }

    getDTGini <- function(data, categ){
        my_method <- "class"
        if (modelType == "regression"){
            my_method <- "anova"
        }
        my_model <- rpart::rpart(as.formula(paste(categ, "~ .")), data = data, method = my_method)
        orig_importance <- my_model$variable.importance
        orig_importance <- orig_importance[setdiff(colnames(data), categ)]
        orig_importance[is.na(orig_importance)] <- 0
        return(orig_importance)
    }
    getDTShapValues <- function(data, categ){
        my_method <- "class"
        if (modelType == "regression"){
            my_method <- "anova"
        }
        my_model <- rpart::rpart(as.formula(paste(categ, "~ .")), data = data, method = my_method)
        shap_values <- fastshap::explain(my_model, X = data[, setdiff(colnames(data), categ)], pred_wrapper = predict, nsim = shap_nsim)
        shap_values <- colMeans(abs(shap_values))
        return(shap_values)
    }

    getRFShapValues <- function(model, data, predictor_names){
        if (modelType == "regression"){
            shap_values <- fastshap::explain(model, X = data[, predictor_names], pred_wrapper = predict, nsim = shap_nsim)
        }
        if (modelType == "classification"){
            pred_wrapper <- function(object, newdata) {
                predict(object, newdata = newdata, type = "prob")[, "A"]
            }
            shap_values <- fastshap::explain(model, X = data[, predictor_names], pred_wrapper = pred_wrapper, nsim = shap_nsim)
        }
        shap_values <- colMeans(abs(shap_values))
        return(shap_values)
    }

    getKerasOldenImportance <- function(model, data){
        # Extraire les poids du modèle Keras
        weights <- model %>% get_weights()

        W1 <- weights[[1]]
        W2 <- weights[[3]]
        W3 <- weights[[5]]

        n_features <- dim(W1)[1]
        n_hidden1 <- dim(W1)[2]  # 16
        n_hidden2 <- dim(W2)[2]  # 8
        n_outputs <- dim(W3)[2]  # num_classes

        feature_importance <- rep(0, n_features)

        for (i in 1:n_features) {
            for (j in 1:n_hidden1) {
                for (k in 1:n_hidden2) {
                    for (l in 1:n_outputs) {
                        feature_importance[i] <- feature_importance[i] + abs(W1[i, j] * W2[j, k] * W3[k, l])
                    }
                }
            }
        }

        names(feature_importance) <- setdiff(colnames(data), categ)

        return(feature_importance)
    }


    getMLPOldenImportance <- function(data){
        library(keras)
        temp_df <- data
        
        cols_to_scale <- setdiff(names(temp_df), categ)
        temp_df[cols_to_scale] <- scale(temp_df[cols_to_scale])

        col_means <- colMeans(temp_df[cols_to_scale])
        col_sds <- apply(temp_df[cols_to_scale], 2, sd)

        n <- nrow(temp_df)
        train_data <- temp_df
        test_data  <- temp_df

        # Préparation des données :
        # Séparer les variables explicatives et la variable cible
        X_train <- as.matrix(train_data[, setdiff(names(train_data), categ)])
        X_test  <- as.matrix(test_data[, setdiff(names(test_data), categ)])

        if (modelType == "classification"){
            # Pour le modèle Keras, il faut encoder la variable cible en one-hot
            num_classes <- length(levels(temp_df[[categ]]))
            # Convertir les classes en indices (commençant par 0)
            y_train <- as.integer(train_data[[categ]]) - 1  
            y_test  <- as.integer(test_data[[categ]]) - 1

            # Encodage one-hot
            y_train <- to_categorical(y_train, num_classes = num_classes)
            y_test  <- to_categorical(y_test, num_classes = num_classes)
        }
        if (modelType == "regression"){
            y_train <- train_data[[categ]]
            y_test  <- test_data[[categ]]
        }

        if (modelType == "classification"){
            model <- keras_model_sequential() %>%
            # Première couche cachée avec 4 neurones et activation ReLU
            layer_dense(units = 16, activation = "relu", input_shape = ncol(X_train)) %>%
            # Dropout classique (20% des neurones désactivés aléatoirement)
            layer_dropout(rate = 0.2) %>%
            # Deuxième couche cachée avec 2 neurones et activation ReLU
            layer_dense(units = 8, activation = "relu") %>%
                layer_dropout(rate = 0.2) %>%
                
            # Deuxième couche cachée avec 2 neurones et activation ReLU

            # Couche de sortie : softmax pour la classification multi-classes
            layer_dense(units = num_classes, activation = "softmax")
        } else {
            model <- keras_model_sequential() %>%
            layer_dense(
                units = 32, 
                activation = "relu", 
                input_shape = ncol(X_train),
                kernel_regularizer = regularizer_l2(l = 0.001)  # Régularisation L2
            ) %>%
            layer_dropout(rate = 0.2) %>%
            
            layer_dense(
                units = 16, 
                activation = "relu",
                kernel_regularizer = regularizer_l2(l = 0.001)  # Régularisation L2
            ) %>%
            
            # Couche de sortie pour la régression (1 neurone, activation linéaire par défaut)
            layer_dense(units = 1)
        }

        if (modelType == "classification"){
            my_loss <- "categorical_crossentropy"
            model %>% compile(
                loss = my_loss,
                optimizer = optimizer_adam(),
                metrics = c("accuracy")
            )
        }
        if (modelType == "regression"){
            my_loss <- "mean_squared_error"
                model %>% compile(
                    loss = my_loss,
                    optimizer = optimizer_adam(learning_rate = 0.01),
                    metrics = c("mean_absolute_error")
                )
        }

        model %>% compile(
            loss = my_loss,
            optimizer = optimizer_adam(),
            metrics = c("accuracy")
        )

        # summary(model)

        history <- model %>% fit(
            X_train, y_train,
            epochs = 5000,          # ajustez le nombre d'epochs selon vos besoins
            #epochs = 100, 
            batch_size = 32,       # ajustez la taille de batch
            validation_split = 0.2,
            verbose = 0
        )

        # Évaluation sur l'ensemble de test
        #score <- model %>% evaluate(X_test, y_test_cat)
        #score
        my_res <- getKerasOldenImportance(model, data)
        return(my_res)
    }

    getMLPShapValues <- function(data){
        library(keras)
        temp_df <- data
        
        cols_to_scale <- setdiff(names(temp_df), categ)
        temp_df[cols_to_scale] <- scale(temp_df[cols_to_scale])

        col_means <- colMeans(temp_df[cols_to_scale])
        col_sds <- apply(temp_df[cols_to_scale], 2, sd)

        n <- nrow(temp_df)
        train_data <- temp_df
        test_data  <- temp_df

        # Préparation des données :
        # Séparer les variables explicatives et la variable cible
        X_train <- as.matrix(train_data[, setdiff(names(train_data), categ)])
        X_test  <- as.matrix(test_data[, setdiff(names(test_data), categ)])

        # Pour le modèle Keras, il faut encoder la variable cible en one-hot
        if (modelType == "classification"){
            num_classes <- length(levels(temp_df[[categ]]))
            # Convertir les classes en indices (commençant par 0)
            y_train <- as.integer(train_data[[categ]]) - 1  
            y_test  <- as.integer(test_data[[categ]]) - 1

            # Encodage one-hot
            y_train <- to_categorical(y_train, num_classes = num_classes)
            y_test  <- to_categorical(y_test, num_classes = num_classes)
        }
        if (modelType == "regression"){
            y_train <- train_data[[categ]]
            y_test  <- test_data[[categ]]
        }

        # Construction du modèle MLP
        if (modelType == "classification"){
            model <- keras_model_sequential() %>%
            # Première couche cachée avec 4 neurones et activation ReLU
            layer_dense(units = 16, activation = "relu", input_shape = ncol(X_train)) %>%
            # Dropout classique (20% des neurones désactivés aléatoirement)
            layer_dropout(rate = 0.2) %>%
            # Deuxième couche cachée avec 2 neurones et activation ReLU
            layer_dense(units = 8, activation = "relu") %>%
                layer_dropout(rate = 0.2) %>%
                
            # Deuxième couche cachée avec 2 neurones et activation ReLU

            # Couche de sortie : softmax pour la classification multi-classes
            layer_dense(units = num_classes, activation = "softmax")
        } else {
            model <- keras_model_sequential() %>%
            layer_dense(
                units = 32, 
                activation = "relu", 
                input_shape = ncol(X_train),
                kernel_regularizer = regularizer_l2(l = 0.001)  # Régularisation L2
            ) %>%
            layer_dropout(rate = 0.2) %>%
            
            layer_dense(
                units = 16, 
                activation = "relu",
                kernel_regularizer = regularizer_l2(l = 0.001)  # Régularisation L2
            ) %>%
            
            # Couche de sortie pour la régression (1 neurone, activation linéaire par défaut)
            layer_dense(units = 1)
        }

        if (modelType == "classification"){
            my_loss <- "categorical_crossentropy"
            model %>% compile(
                loss = my_loss,
                optimizer = optimizer_adam(),
                metrics = c("accuracy")
            )
        }
        if (modelType == "regression"){
            my_loss <- "mean_squared_error"
            model %>% compile(
                loss = my_loss,
                optimizer = optimizer_adam(learning_rate = 0.01),
                metrics = c("mean_absolute_error")
            )
        }


        # summary(model)

        history <- model %>% fit(
            X_train, y_train,
            epochs = 5000,        # ajustez le nombre d'epochs selon vos besoins
            #epochs = 100, 
            batch_size = 32,       # ajustez la taille de batch
            validation_split = 0.2,
            verbose = 0
        )
        pred_wrapper <- function(object, newdata, ...) {
            predict(object, newdata)[, 1]
        }
        shap_values <- fastshap::explain(object=model, X = X_train, pred_wrapper = pred_wrapper, nsim = shap_nsim)
        shap_values <- colMeans(abs(shap_values))
        return(shap_values)
    }

    getSVMShapValues <- function(data){
        str_kern <- unlist(strsplit(ml_algo, 'svm-'))[2]
        #str_kern <- "rbfdot"
        svm_model_ksvm <- kernlab::ksvm(as.formula(paste(categ, "~ .")), 
                       data = data, 
                       kernel = str_kern,   # Noyau polynomial rbfdot polydot
                       prob.model = TRUE)
        X <- data[, setdiff(colnames(data), categ)]
        pred_fun <- function(model, newdata) {
            probs <- kernlab::predict(model, newdata, type = "probabilities")
            return(probs[, 1])
        }
        shap_values <- fastshap::explain(svm_model_ksvm, 
                                        X = X, 
                                        pred_wrapper = pred_fun, 
                                        nsim = shap_nsim)
        shap_importance <- colMeans(abs(shap_values))
    }

    getSVMCoeffValues <- function(data){
        str_kern <- unlist(strsplit(ml_algo, 'svm-'))[2]
        #str_kern <- "rbfdot"
        svm_model_ksvm <- kernlab::ksvm(as.formula(paste(categ, "~ .")), 
                       data = data, 
                       kernel = str_kern,   # Noyau polynomial rbfdot polydot
                       prob.model = TRUE)
                       # Extraction des coefficients (alpha) pour le cas binaire
        alphas <- coef(svm_model_ksvm)

        # Extraction des vecteurs de support et conversion en matrice
        SV <- do.call(rbind, svm_model_ksvm@xmatrix)

        # Calcul du vecteur de poids (w)
        w <- t(alphas) %*% SV

        # Pour obtenir une importance relative des features,
        # on peut utiliser la valeur absolue des coefficients:
        feature_importance <- as.vector(abs(w))

        # Affichage du vecteur d'importance

        # Optionnel : association aux noms des colonnes (features)
        names(feature_importance) <- colnames(SV)
        return(feature_importance)
    }

    start_time <- Sys.time()
    
    if (ml_algo == "mlp"){
        if (fi_algo == "olden"){
            orig_importance <- getMLPOldenImportance(data)
        }
        if (fi_algo == "shap"){
            orig_importance <- getMLPShapValues(data)
        }
        if ( fi_algo == "lime"){
            orig_importance <- featureImportanceLime(data, y=categ, ml_algo=ml_algo)
        }
    }

    if (ml_algo == "rf"){
        if (fi_algo != "lime"){
            my_model <- randomForest::randomForest(as.formula(paste(categ, "~ .")), data = data, importance = TRUE)
        
        if (fi_algo %in% c("gini", "accuracy")){
            my_type <- 1
            if (fi_algo == "gini"){
                my_type <- 2
            }
            orig_importance <- randomForest::importance(my_model, type = my_type)[, 1]
        }   
        if (fi_algo == "shap"){
            orig_importance <- getRFShapValues(my_model, data, predictor_names)
        }
        } else {
            if (fi_algo == "lime"){
                orig_importance <- featureImportanceLime(data, y=categ, ml_algo=ml_algo, modelType=modelType)
            }
        }
    }

    if (ml_algo == "dt"){
        if (fi_algo == "gini"){
            orig_importance <- getDTGini(data, categ)
        }
        if (fi_algo == "shap"){
            orig_importance <- getDTShapValues(data, categ)
        }
        if (fi_algo == "lime"){
            orig_importance <- featureImportanceLime(data, y=categ, ml_algo=ml_algo, modelType=modelType)
        }
    }

    if (unlist(strsplit(ml_algo, '-'))[1] == "svm"){
        if (fi_algo == "shap"){
            orig_importance <- getSVMShapValues(data)
        }
        if(fi_algo == "coeff"){
            orig_importance <- getSVMCoeffValues(data)
        }
        if (fi_algo == "lime"){
            orig_importance <- featureImportanceLime(data, y=categ, ml_algo=ml_algo, modelType=modelType)
        }
    }

    if (verbose>0){
        cat("Observed importance:\n")
        print(orig_importance)
    }

    if (fi_transfo == "rank" || fi_transfo == "rank+jitter" || fi_transfo == "jitter"){
        temp_names <- names(orig_importance)
        #orig_importance <- scale(orig_importance)
        if (fi_transfo == "rank+jitter" || fi_transfo == "jitter"){
            toadd <- ((runif(length(orig_importance))-0.5)*jitter_ratio*orig_importance)
            toadd[toadd==0] <- jitter_ratio * min(abs(orig_importance[orig_importance!=0])) * (runif(sum(toadd==0))-0.5)
            orig_importance <- orig_importance + toadd
        }
        if(fi_transfo != "jitter"){
            orig_importance <- rank(abs(orig_importance))
            # orig_importance <- abs(orig_importance)
        }
        names(orig_importance) <- temp_names
        
    }

    if (verbose>0){
        cat("Observed importance:\n")
        print(orig_importance)
    }

    end_time <- Sys.time()
    computeTimes[["diff_time1"]] <-  difftime(end_time, start_time, units="secs")
    
    start_time <- Sys.time()
    if (pv_algo == "probes"){
        n_perm <- n_perm   # Nombre de répétitions (permutations)
        n_probes <- n_probes   # Nombre de colonnes probes à ajouter par permutation
        # Vecteur pour stocker toutes les importances obtenues pour les probes
        #probe_importances <- c()
        df_probe_importances <- as.data.frame(matrix(nrow = 0, ncol = length(orig_importance)))
        colnames(df_probe_importances) <- names(orig_importance)
        # Pour chaque permutation, on ajoute nprobes colonnes aléatoires et on ajuste le modèle
        # display(paste0("Permutation ", 0, " sur ", n_perm))
        for(i in 1:n_perm){
            #display(paste0("Permutation ", i, " sur ", n_perm))
            # Copier le jeu de données original
            if (verbose>0){
                cat("Probes permutation ", i, " sur ", n_perm, "\n")
            }
            #message("Probes permutation ", i, " sur ", n_perm)
            data_perm <- data
            # Ajouter nprobes colonnes de bruit (probes)
            for(j in 1:n_probes){
                probe_name <- paste0("probe_", j)
                data_perm[[probe_name]] <- rnorm(nrow(data))  # bruit normal standard
            }
            # Ajuster un modèle sur le jeu de données étendu
            if (ml_algo == "rf"){
                if (fi_algo != "lime"){
                    model_perm <- randomForest::randomForest(as.formula(paste(categ, "~ .")), data = data_perm, importance = TRUE)
                    if (fi_algo %in% c("gini", "accuracy")){
                        imp_perm <- randomForest::importance(model_perm, type = my_type)[,1]
                    }
                    if (fi_algo == "shap"){
                        imp_perm <- getRFShapValues(model_perm, data_perm, setdiff(colnames(data_perm), categ))
                    }
                } else {
                    if (fi_algo == "lime"){
                        imp_perm <- featureImportanceLime(data_perm, y=categ, ml_algo=ml_algo, modelType=modelType)
                    }
                }
            }
            if (ml_algo == "dt"){
                if (fi_algo == "gini"){
                    imp_perm <- getDTGini(data_perm, categ)
                }
                if (fi_algo == "shap"){
                    imp_perm <- getDTShapValues(data_perm, categ)
                }
                if (fi_algo == "lime"){
                    imp_perm <- featureImportanceLime(data_perm, y=categ, ml_algo=ml_algo, modelType=modelType)
                }
            }
                    
            if (ml_algo == "mlp"){
                if (fi_algo == "olden"){
                    imp_perm <- getMLPOldenImportance(data_perm)
                }
                if (fi_algo == "shap"){
                    imp_perm <- getMLPShapValues(data_perm)
                }
                if (fi_algo == "lime"){
                    imp_perm <- featureImportanceLime(data_perm, y=categ, ml_algo=ml_algo, modelType=modelType)
                }
            }
            if (unlist(strsplit(ml_algo, '-'))[1] == "svm"){
                if (fi_algo == "shap"){
                    imp_perm <- getSVMShapValues(data_perm)
                }
                if(fi_algo == "coeff"){
                    imp_perm <- getSVMCoeffValues(data_perm)
                }
                if (fi_algo == "lime"){
                    imp_perm <- featureImportanceLime(data_perm, y=categ, ml_algo=ml_algo, modelType=modelType)
                }
            }
            #if (fi_algo == "olden" || fi_algo == "shap"){
            if (fi_transfo == "rank" || fi_transfo == "rank+jitter" || fi_transfo == "jitter"){
                temp_names <- names(imp_perm)
                #imp_perm <- scale(imp_perm)[,1]
                if (fi_transfo == "rank+jitter" || fi_transfo == "jitter"){
                    toadd <- ((runif(length(imp_perm))-0.5)*jitter_ratio*imp_perm)
                    toadd[toadd==0] <- jitter_ratio * min(abs(imp_perm[imp_perm!=0])) * (runif(sum(toadd==0))-0.5)
                    imp_perm <- imp_perm + toadd
                }
                if(fi_transfo != "jitter"){
                    imp_perm <- rank(abs(imp_perm))-n_probes
                    # imp_perm <- abs(imp_perm)
                }
                names(imp_perm) <- temp_names
            }

            # Conserver uniquement les colonnes probes (leurs noms commencent par "probe_")
            probe_imp_vals <- imp_perm[grep("probe_", names(imp_perm))]
            # Ajouter ces importances au vecteur global
            #probe_importances <- c(probe_importances, probe_imp_vals)

            p_values <- sapply(names(orig_importance), function(feat) {
                (1 + sum(probe_imp_vals >= imp_perm[feat])) / (length(probe_imp_vals) + 1)
            })
            df_probe_importances <- rbind(df_probe_importances, p_values)
            # names(p_values) <- names(orig_importance)
        }

        #p_values <- sapply(names(orig_importance), function(feat) {
        #    (1 + sum(probe_importances >= orig_importance[feat])) / (length(probe_importances) + 1)
        #})
        #names(p_values) <- names(orig_importance)

    # Calculate p-values as the column means of df_probe_importances
        if (nrow(df_probe_importances) > 0) {
            p_values <- colMeans(df_probe_importances, na.rm = TRUE)
            names(p_values) <- names(orig_importance)
        }
    }

    if (pv_algo == "pimp"){
        # 2. Estimate the null distribution by permuting the target variable
        n_perm <- n_perm  # number of permutations (increase for more robustness)
        # Determine predictor names (all columns except the target)
        if (verbose > 0){
            cat("Predictor names: ", predictor_names, "\n")
        }

        p <- length(predictor_names)
        importance_null <- matrix(NA, nrow = n_perm, ncol = p)
        colnames(importance_null) <- predictor_names
        for (i in 1:n_perm) {
            data_perm <- data
            # Permute the target variable to break the relationship between predictors and target
            if (verbose>0){
                cat("Pimp permutation ", i, " sur ", n_perm, "\n")
            }
            #message("Pimp permutation ", i, " sur ", n_perm)
            data_perm[[categ]] <- sample(data_perm[[categ]])
            if (ml_algo == "mlp"){
                if (fi_algo == "olden"){
                    importance_null[i, ] <- getMLPOldenImportance(data_perm)
                }
                if (fi_algo == "shap"){
                    importance_null[i, ] <- getMLPShapValues(data_perm)
                }
                if (fi_algo == "lime"){
                    importance_null[i, ] <- featureImportanceLime(data_perm, y=categ, ml_algo=ml_algo, modelType=modelType)
                }
            }
            if (ml_algo == "rf"){
                if (fi_algo != "lime"){
                    model_perm <- randomForest::randomForest(as.formula(paste(categ, "~ .")), data = data_perm, importance = TRUE)
                    if (fi_algo %in% c("gini", "accuracy")){
                        importance_null[i, ] <- randomForest::importance(model_perm, type = my_type)[, 1]
                    }
                    if (fi_algo == "shap"){
                        importance_null[i, ] <- getRFShapValues(model_perm, data_perm, setdiff(colnames(data_perm), categ))
                    }
                } else {
                    if (fi_algo == "lime"){
                        importance_null[i, ] <- featureImportanceLime(data_perm, y=categ, ml_algo=ml_algo, modelType=modelType)
                    }
                }

            }
            if (ml_algo == "dt"){
                if (fi_algo == "gini"){
                    importance_null[i, ] <- getDTGini(data_perm, categ)
                }
                if (fi_algo == "shap"){
                    importance_null[i, ] <- getDTShapValues(data_perm, categ)
                }
                if (fi_algo == "lime"){
                    importance_null[i, ] <- featureImportanceLime(data_perm, y=categ, ml_algo=ml_algo, modelType=modelType)
                }
            }
            if (unlist(strsplit(ml_algo, '-'))[1] == "svm"){
                if (fi_algo == "shap"){
                    importance_null[i, ] <- getSVMShapValues(data_perm)
                }
                if(fi_algo == "coeff"){
                    importance_null[i, ] <- getSVMCoeffValues(data_perm)
                }
                if (fi_algo == "lime"){
                    importance_null[i, ] <- featureImportanceLime(data_perm, y=categ, ml_algo=ml_algo, modelType=modelType)
                }
            }
        }
        # 3. Calculate p-values: proportion of permutations where the null importance exceeds the observed importance
        #display(importance_null)
        #display(orig_importance)
        if (fi_transfo == "rank" || fi_transfo == "rank+jitter" || fi_transfo == "jitter"){
            temp_names <- colnames(importance_null)
            importance_null <- t(apply(importance_null, 1, function(x) {
                #scale(abs(x))
                y <- x
                #display("orig x")
                #display(x)
                if (fi_transfo == "rank+jitter" || fi_transfo == "jitter"){
                    toadd <- ((runif(length(x))-0.5)*jitter_ratio*x)
                    toadd[toadd==0] <- jitter_ratio * min(abs(x[x!=0])) * (runif(sum(toadd==0))-0.5)
                    y <- y + toadd
                }
                if(fi_transfo != "jitter"){
                    y <- rank(abs(y))
                }
                return(y)
            }))
            colnames(importance_null) <- temp_names
        }
        #display(importance_null)
        #display(orig_importance)
        #display("importance_null[,'diff_distrib02']")
        #display(importance_null[,"diff_distrib02"])
        #display("orig_importance['diff_distrib02']")
        #display(orig_importance["diff_distrib02"])
        
        p_values <- sapply(1:p, function(j) {
            if ( is.na(orig_importance[j]) ){
                return(1)
            }
            (1 + sum(importance_null[, j] >= orig_importance[j])) / (n_perm + 1)
        })
        names(p_values) <- predictor_names

        if ( verbose > 0){
            cat("\nP-values obtained by the PIMP method:\n")
            print(p_values)
        }
    }

    cols2keep <- setdiff(colnames(data), categ)
    p_values <- p_values[cols2keep]

    if (verbose > 0){
        print("p_values")
        print(p_values)
    }
    adjPvals <- p.adjust(p_values, method = adjMethod)
    featImps <- orig_importance[cols2keep]
    featImps[is.na(featImps)] <- 0

    end_time <- Sys.time()
    computeTimes[["diff_time2"]] <-  difftime(end_time, start_time, units="secs")

    my_res <- list( featImps=orig_importance[cols2keep],
                    pvals=p_values,
                    adjPvals=adjPvals,
                    computeTimes=computeTimes)
    return(my_res)
}
