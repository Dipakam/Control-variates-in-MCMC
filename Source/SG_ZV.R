source('zv.R')
require(mvtnorm)

# gradient of the log-likelihood of the logit model.
f.vanilla.gradient <- function(y, X, beta)
{
  n <- dim(X)[1]
  beta <- matrix(beta, ncol = 1)
  p_i <- exp(X %*% beta) / (1 + exp(X%*%beta))  
  rtn <- colSums(X* as.numeric(y - p_i))
  return(n*rtn)
}

# Identity function
id <- function(theta){
  return (theta)
}

#### Stochastic Gradient #####
SGA.vanilla <- function(y, X, batch.size = dim(X)[1], t = .1, max.iter = dim(X)[1], adapt = FALSE)
{  
  p <- dim(X)[2]
  n <- dim(X)[1]
  
  #   create the mini-batches
  permutation <- sample(1:n, replace = FALSE)
  K <- floor(n/batch.size)
  batch.index <- split(permutation, rep(1:K, length = n, each = n/K))
  
  #   index for choosing the mini-batch
  count <- 1
  
  #Initialisation of output matrices
  beta_k <- rep(0, p) # st at all 0s
  track.gradient <- matrix(0, nrow = max.iter, ncol = p)
  track.beta <- matrix(0, nrow = max.iter, ncol = p)
  
  #   saving the running mean of the estimates of theta^*
  mean_beta <- coeff #rep(0,p)
  
  # tk: in case we want t_k
  tk <- t

  for(iter in 1:max.iter)  
  {
    count <- count+1
    if(count %% K == 0) count <- count%%K  +1  # when all batches finish, restart the batches
    
    # batch of data
    y.batch <- y[batch.index[[count]] ]
    X.batch <- matrix(X[batch.index[[count]], ], nrow = batch.size)
    
    # SGA step
    track.gradient[iter,] <- f.vanilla.gradient(y = y.batch, X = X.batch, beta = beta_k)/batch.size
    beta_k = beta_k + tk*track.gradient[iter,] 
    
    est = beta_k
    track.beta[iter, ] <- est
  }
  rtn <- list("iter" = iter, "est" = est, "iterates" = track.beta[1:iter, ], "grad" = track.gradient[1:iter,])
  return(rtn)
}


# Generating Data
set.seed(12)
p <- 30
n <- 1e6
X <- matrix(rnorm(n*(p)), nrow = n, ncol = p)
beta <- matrix(rnorm(p, 0, sd = 3), ncol = 1)
p_i <- exp(X %*% beta)/(1 + exp(X%*%beta))
y <- rbinom(n, size = 1, prob = p_i)

# True MLE
coeff <- glm(y ~ X-1,family = "binomial")$coeff

# Test for variance reduction.
n.samp = 50

org.est = matrix(0, nrow = n.samp, ncol = p)
mod.est = matrix(0, nrow = n.samp, ncol = p)

for (i in 1:n.samp){
  b10 <- SGA.vanilla(y, X, batch.size = 100, t = .01, max.iter = 1e5)
  
  #   Apply ZV postprocessing
  original.iter = t(b10$iterates)
  modified.iter = postprocess.zv(original.iter, t(b10$grad), id)
  
  #   Save estimates
  org.est[i, ] <- t(as.matrix(b10$est))
  mod.est[i, ] <- t(as.matrix(rowMeans(modified.iter)))
  print(i)
}

var.red = mean(sqrt(apply(org.est, 2, var)/apply(mod.est, 2, var))) # This value should be > 1 for success.

apply(org.est, 2, mean) # Check if mean is close enough to true MLE
apply(mod.est, 2, mean) # Check if mean is close enough to true MLE

# Output Analysis
table = matrix(NA, nrow = 5, ncol = 6)

for (i in 1:5)
{
  sd_org = sd(org.est[, i])
  sd_mod = sd(mod.est[, i])
  table[i, ] = c(coeff[i], mean(org.est[, i]), mean(mod.est[, i]), 
                 sd_org, sd_mod, sd_org/sd_mod)
}
colnames(table) <- c("True Value", "Original Estimate", "Postprocessed Estimate", 
                     "SD Original", "SD Postprocessed", "Ratio")


grDevices::pdf("./Plots/ZV_tsplots.pdf", width = 8, height = 8)
index = 1:n.samp
par(mfrow = c(2, 2))
for (i in 1:4){
  plot.ts(org.est[, i], ylab = "Mean Estimates", xlab = "Index")
  lines(index, mod.est[, i], col = "blue")
  abline(h = coeff[i], col = "green")
  legend("bottomright", legend = c("Standard", "ZV Processed", "True Value"), 
         col = c("black", "blue", "green"), lty = c(1, 1, 1), cex = 0.50)
}
dev.off()
