##### MALA #####
# Log-likelihood of logit model.
log.target <- function(beta, y. = y, X. = X)
{
  one.minus.yx <- (1 - y.)*X.
  ret = (-sum(log(1 + exp(-X. %*% as.matrix(beta, ncol = 1)))) - sum(one.minus.yx%*%beta))
  return (ret)
}

# Gradient of the log-likelihood
grad.target <- function(beta, y. = y, X. = X)
{
  beta <- matrix(beta, ncol = 1)
  p_i <- exp(X. %*% beta) / (1 + exp(X. %*% beta))  
  rtn <- colSums(X. * as.numeric(y. - p_i))
  return(n*rtn)
}

# Identity function
id <- function(theta){
  return (theta)
}

MALA <- function(h = 1, batch.size = dim(X)[1], niter = 5000){
  # create the mini-batches
  permutation <- sample(1:n, replace = FALSE)
  K <- floor(n/batch.size)
  batch.index <- split(permutation, rep(1:K, length = n, each = n/K))
  
  # Initialisation of output matrices
  st = rep(0, p) # Start at all 0s
  len = p
  result = matrix(NA, niter, len)
  track.gradient = matrix(NA, niter, len)
  acc = 0
  count = 0
  
  x = st
  for (i in 1:niter){
    count = count + 1
    if((count-1) %% K == 0) count <- 1  # when all batches finish, restart the batches
    
    # batch of data
    y.batch <- y[batch.index[[count]] ]
    X.batch <- matrix(X[batch.index[[count]], ], nrow = batch.size)
    
    # MALA proposal
    mu.x = x + 0.5*h*grad.target(x, y. = y.batch, X. = X.batch)/batch.size
    z = rnorm(len, mean = mu.x, sd = sqrt(h))
    mu.z = z + 0.5*h*grad.target(z, y. = y.batch, X. = X.batch)/batch.size
    
    # Accept or not to accept
    alpha = exp(log.target(z) - log.target(x)
                + dmvnorm(x, mean = mu.z, sigma = h*diag(len), log = TRUE)
                - dmvnorm(z, mean = mu.x, sigma = h*diag(len), log = TRUE))
    
    if (runif(1) < alpha){
      x = z
      acc = acc + 1
    }
    result[i, ] = x
    track.gradient[i, ] = grad.target(x, y. = y.batch, X. = X.batch)/batch.size
  }
  print (acc/niter)
  return (list("iterates" = result, "grad" = track.gradient))
}

# Generating Data
set.seed(12)
p <- 30
n <- 1e6
X <- matrix(rnorm(n*(p)), nrow = n, ncol = p)
# X <- cbind(1, X)
beta <- matrix(rnorm(p, 0, sd = 3), ncol = 1)
p_i <- exp(X %*% beta)/(1 + exp(X%*%beta))
y <- rbinom(n, size = 1, prob = p_i)

# True MLE
coeff <- glm(y ~ X-1,family = "binomial")$coeff

# Hyper-parameters
h = 0.007
niter = 100000

# Test for variance reduction
n.samp = 20
org.est = matrix(NA, nrow = n.samp, ncol = p)
mod.est = matrix(NA, nrow = n.samp, ncol = p)

for (i in 1:n.samp){
  mala.chain <- MALA(h, batch.size = 10,  niter)
  
  #Apply ZV postprocessing
  original.iter = t(mala.chain$iterates)
  modified.iter = postprocess.zv(original.iter, t(mala.chain$grad), id)
  
  # Save estimates
  org.est[i, ] <- apply(original.iter, 1, mean)
  mod.est[i, ] <- apply(modified.iter, 1, mean)
  print(i)
}

var.red = apply(org.est, 2, var, na.rm = TRUE)/apply(mod.est, 2, var, na.rm = TRUE) # var.red > 1 -> Successful

apply(org.est, 2, mean, na.rm = TRUE) # Check if mean is close enough to true MLE
apply(mod.est, 2, mean, na.rm = TRUE) # Check if mean is close enough to true MLE

# Output Analysis
grDevices::pdf("./Plots/MALAZV_tsplots.pdf", width = 8, height = 8)
index = 1:n.samp
par(mfrow = c(2, 2))
for (i in 1:4){
  low= min(mod.est[, i], org.est[, i], coeff[i]) - 0.2
  up = max(mod.est[, i], org.est[, i], coeff[i]) + 0.2
  plot.ts(mod.est[, i], col = "blue", ylab = "Mean Estimates", xlab = "Index", ylim = c(low,up))
  lines(index, org.est[, i], col = "black")
  abline(h = coeff[i], col = "green")
  legend("bottomright", legend = c("Standard", "ZV Processed", "True Value"), 
         col = c("black", "blue", "green"), lty = c(1, 1, 1), cex = 0.65)
}
dev.off()