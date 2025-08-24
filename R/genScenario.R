genScenario <- function(scenario=1,
                            n_samples = 100, 
                            n_norm_noise_features = NULL,
                            n_unif_noise_features = NULL){
    # Samples
    if (is.null(n_norm_noise_features)){
        n_norm_noise_features <- 10
    }
    if (is.null(n_unif_noise_features)){
        if (scenario %in% c(1,2,3,4)){
            n_unif_noise_features <- 0
        }
        if (scenario %in% c(5,6)){
            n_unif_noise_features <- 10
        }
    }

    half_samples <- floor(n_samples/2)
    remaining_samples <- n_samples - half_samples
    df_simu = data.frame(y=c(rep("A", half_samples), rep("B", remaining_samples)))
    # Noise columns

    noise_columns <- paste0("norm_noise", sprintf("%02d", 1:n_norm_noise_features))
    df_simu[noise_columns] <- lapply(noise_columns, function(column_name) {
        mean <- runif(1, min = -10, max = 10)
        sd <- runif(1, min = 1, max = 5)
        rnorm(nrow(df_simu), mean = mean, sd = sd)
    })

    unif_noise_columns <- paste0("unif_noise", sprintf("%02d", 1:n_unif_noise_features))
    df_simu[unif_noise_columns] <- lapply(unif_noise_columns, function(column_name) {
        rand_min <- runif(1, min = -10, max = 0)
        rand_max <- runif(1, min = 0, max = 10)
        runif(nrow(df_simu), min = rand_min, max = rand_max)
    })

    if (scenario == 1){
        # Distribution columns
        rand_mean = 2
        rand_sd = 2
        mask_group = df_simu[["y"]] == "A"
        df_simu[["diff_distrib01"]][mask_group] <- rnorm(sum(mask_group), mean = rand_mean, sd = rand_sd)
        mask_group = df_simu[["y"]] == "B"
        df_simu[["diff_distrib01"]][mask_group] <- rnorm(sum(mask_group), mean = -rand_mean, sd = rand_sd)

        rand_mean = 4
        rand_sd = 4
        mask_group = df_simu[["y"]] == "A"
        df_simu[["diff_distrib02"]][mask_group] <- rnorm(sum(mask_group), mean = rand_mean, sd = rand_sd)
        mask_group = df_simu[["y"]] == "B"
        df_simu[["diff_distrib02"]][mask_group] <- rnorm(sum(mask_group), mean = -rand_mean, sd = rand_sd)

        rand_mean = 1
        rand_sd1 = 1
        rand_sd2 = 5
        mask_group = df_simu[["y"]] == "A"
        df_simu[["diffVar01"]][mask_group] <- rnorm(sum(mask_group), mean = rand_mean, sd = rand_sd1)
        mask_group = df_simu[["y"]] == "B"
        df_simu[["diffVar01"]][mask_group] <- rnorm(sum(mask_group), mean = -rand_mean, sd = rand_sd2)    
    }

    if (scenario == 2){
        rand_mean = 3
        rand_sd = 2
        mask_group = df_simu[["y"]] == "A"
        df_simu[["diff_distrib01"]][mask_group] <- rnorm(sum(mask_group), mean = rand_mean, sd = rand_sd)
        mask_group = df_simu[["y"]] == "B"
        df_simu[["diff_distrib01"]][mask_group] <- rnorm(sum(mask_group), mean = -rand_mean, sd = rand_sd)

        rand_mean = 5
        rand_sd = 2
        mask_group = df_simu[["y"]] == "A"
        df_simu[["diff_distrib02"]][mask_group] <- rnorm(sum(mask_group), mean = rand_mean, sd = rand_sd)
        rand_mean = -4
        rand_sd = 4
        mask_group = df_simu[["y"]] == "B"
        df_simu[["diff_distrib02"]][mask_group] <- rnorm(sum(mask_group), mean = rand_mean, sd = rand_sd)

        c_x1 <- rnorm(n = 25, mean = 1, sd = 1)
        c_x2 <- rnorm(n = 25, mean = 11, sd = 2)
        c_x <- c(c_x1, c_x2)
        c_y <- rnorm(n = 50, mean = 5, sd = 1)
        df_simu$biDistrib = c(c_x, c_y)
    }

    if (scenario == 3){
        rand_mean = 2
        rand_sd = 2
        mask_group = df_simu[["y"]] == "A"
        df_simu[["diff_distrib01"]][mask_group] <- rnorm(sum(mask_group), mean = rand_mean, sd = rand_sd)
        mask_group = df_simu[["y"]] == "B"
        df_simu[["diff_distrib01"]][mask_group] <- rnorm(sum(mask_group), mean = -rand_mean, sd = rand_sd)

        rand_mean = 4
        rand_sd = 4
        mask_group = df_simu[["y"]] == "A"
        df_simu[["diff_distrib02"]][mask_group] <- rnorm(sum(mask_group), mean = rand_mean, sd = rand_sd)
        mask_group = df_simu[["y"]] == "B"
        df_simu[["diff_distrib02"]][mask_group] <- rnorm(sum(mask_group), mean = -rand_mean, sd = rand_sd)

        mask_group <- 1:25
        df_simu[['relVar01']][mask_group] <- rnorm(length(mask_group), mean = 2, sd = 1)
        df_simu[['relVar02']][mask_group] <- rnorm(length(mask_group), mean = 2, sd = 1)

        mask_group <- 26:50
        df_simu[['relVar01']][mask_group] <- rnorm(length(mask_group), mean = -2, sd = 1)
        df_simu[['relVar02']][mask_group] <- rnorm(length(mask_group), mean = -2, sd = 1)

        mask_group <- 51:75
        df_simu[['relVar01']][mask_group] <- rnorm(length(mask_group), mean = -2, sd = 1)
        df_simu[['relVar02']][mask_group] <- rnorm(length(mask_group), mean = 2, sd = 1)

        mask_group = 76:100
        df_simu[['relVar01']][mask_group] <- rnorm(length(mask_group), mean = 2, sd = 1)
        df_simu[['relVar02']][mask_group] <- rnorm(length(mask_group), mean = -2, sd = 1)
    }

    if (scenario == 4){
        rand_mean = 2
        rand_sd = 2
        mask_group = df_simu[["y"]] == "A"
        df_simu[["diff_distrib01"]][mask_group] <- rnorm(sum(mask_group), mean = rand_mean, sd = rand_sd)
        mask_group = df_simu[["y"]] == "B"
        df_simu[["diff_distrib01"]][mask_group] <- rnorm(sum(mask_group), mean = -rand_mean, sd = rand_sd)

        rand_mean = 4
        rand_sd = 4
        mask_group = df_simu[["y"]] == "A"
        df_simu[["diff_distrib02"]][mask_group] <- rnorm(sum(mask_group), mean = rand_mean, sd = rand_sd)
        mask_group = df_simu[["y"]] == "B"
        df_simu[["diff_distrib02"]][mask_group] <- rnorm(sum(mask_group), mean = -rand_mean, sd = rand_sd)

        # Create concentric circles for class A
        n_outer <- 35
        n_inner <- 15

        # Outer circle (radius 4)
        theta_outer <- seq(0, 2*pi, length.out = n_outer)
        outer_x <- 4 * cos(theta_outer) + rnorm(n_outer, mean = 0, sd = 0.2)
        outer_y <- 4 * sin(theta_outer) + rnorm(n_outer, mean = 0, sd = 0.2)

        # Inner circle (radius 0.5) 
        theta_inner <- runif(n_inner, 0, 2*pi)
        inner_x <- 0.5 * cos(theta_inner) + rnorm(n_inner, mean = 0, sd = 0.2)
        inner_y <- 0.5 * sin(theta_inner) + rnorm(n_inner, mean = 0, sd = 0.2)

        # Combine circles for class A
        df_simu[['relVar01']][df_simu$y == "A"] <- c(outer_x, inner_x)
        df_simu[['relVar02']][df_simu$y == "A"] <- c(outer_y, inner_y)

        # Circle for class B (radius 2.5)
        theta_B <- seq(0, 2*pi, length.out = 50)
        df_simu[['relVar01']][df_simu$y == "B"] <- 2.5 * cos(theta_B) + rnorm(50, mean = 0, sd = 0.2)
        df_simu[['relVar02']][df_simu$y == "B"] <- 2.5 * sin(theta_B) + rnorm(50, mean = 0, sd = 0.2)
    }

    if (scenario == 5){
        transfo_parab <- function(xs){
            x1 <- min(xs)
            x2 <- max(xs)
            h <- (x1 + x2) / 2
            k <- max(xs)/2
            a <- k / ((h - x1) * (h - x2))
            
            y <- a * (xs - x1) * (xs - x2)
            
            return (y)
        }

        df_simu$y <- transfo_parab(df_simu$norm_noise01)

    }

    if (scenario == 6){
        croissSinusoid <- function(x){
            return (sin(x) * x)
        }
        df_simu[['unif_noise01']] <- runif(nrow(df_simu), min = 0, max = 12.6)

        df_simu$y <- croissSinusoid(df_simu$unif_noise01)
    }

    return(df_simu)
}
