# The sigmoid function.
sigmoid = function(x) {
  1 / (1 + exp(-x))
}

# Log-likelihood for the logit model
log.logit <- function(beta, y = y.actual, X. = X)
{
  one.minus.yx <- (1 - y)*X.
  ret = (-sum(log(1 + exp(-X. %*% as.matrix(beta, ncol = 1)))) - sum(one.minus.yx%*%beta))
  return (ret)
}

# Gradient of log-likelihood the logit model.
f.gradient <- function(y, X, beta)
{
  n <- dim(X)[1]
  beta <- matrix(beta, ncol = 1)
  pi.mult <- 1 / (1 + exp( (X%*%beta) * as.numeric(y)  ))  
  rtn <- X* as.numeric(y*pi.mult)
  return (rtn)
}


# Compute the expectation of the control variates.
E.hd <- function(y, X, weight){
  
  n <- dim(X)[1]
  X_pos <- X[which(y==1),]
  X_neg <- X[which(y==-1),]
  
  n_pos <- length(X_pos[,1])
  n_neg <- length(X_neg[,1])
  
  X_pos_mean <- colMeans(X_pos)
  X_pos_var <- var(X_pos)
  X_neg_mean <- colMeans(X_neg)
  X_neg_var <- var(X_neg)
  
  z_pos_cap <- as.numeric(-t(weight)%*%X_pos_mean)
  E_pos = sigmoid(z_pos_cap)*(X_pos_mean*(1-sigmoid(-z_pos_cap)*z_pos_cap) - sigmoid(-z_pos_cap)*(X_pos_var + X_pos_mean%*%t(X_pos_mean))%*%weight
  )
  
  z_neg_cap <- as.numeric(t(weight)%*%X_neg_mean)
  E_neg = -sigmoid(z_neg_cap)*(X_neg_mean*(1-sigmoid(-z_neg_cap)*z_neg_cap) + sigmoid(-z_neg_cap)*(X_neg_var + X_neg_mean%*%t(X_neg_mean))%*%weight
  )
  
  return ((n_pos*E_pos + n_neg*E_neg)/n)
}

# Gradient of control variates.
h.gradient <- function(y,X,beta)
{
  
  X_pos <- X[which(y==1),]
  X_neg <- X[which(y==-1),]
  
  n_pos <- length(X_pos[,1])
  n_neg <- length(X_neg[,1])
  
  X_pos_mean <- colMeans(X_pos)
  X_neg_mean <- colMeans(X_neg)

  
  n <- dim(X)[1]
  beta <- matrix(beta, ncol = 1)
  z <- (X%*%beta) * as.numeric(0 - y)
  z1.cap <- as.numeric( (-1)*(X_pos_mean %*% beta))
  z2.cap <- as.numeric( (X_neg_mean %*% beta))
  z.cap <-  (n_pos*z1.cap+n_neg*z2.cap)/n + as.numeric(y * (n_pos*z1.cap - n_neg*z2.cap)/n)
  rtn <- ( X * as.numeric(sigmoid(z.cap) * as.numeric(y)) * as.numeric(1 + sigmoid(0 - z.cap)*(-z.cap - (X %*% beta) * as.numeric(y)) ) )

  return (rtn)
}

# Computing optimal A*
A.opt <- function(y, X, beta){
  h_w <- h.gradient(y, X, beta)
  g_w <- f.gradient(y, X, beta)
  var_h_w = var(h_w)
  cov_gh_w = cov(g_w, h_w)
  a = rep(0, 5)
  for (i in 1:5){
    a[i] = cov_gh_w[i, i]/var_h_w[i, i]
  }
  rtn = diag(a)
  return (rtn)
}

# Final CV gradient (incl. standard gradient)
cv.gradient <- function(y, X, beta){
  g_w <- f.gradient(y, X, beta)
  A_w <- A.opt(y,X,beta)
  h_w <- h.gradient(y, X, beta)
  E_w <- E.hd(y, X, beta)
  
  g_w_cv <- colMeans(g_w) + t(A_w)%*%(colMeans(h_w) - E_w)
  
  return (g_w_cv)
}
################################################
## MLE for logistic regression
## Using stochastic gradient ascent
################################################
# Gradient of log-likelihood of logit model for SG
f_vanilla.gradient <- function(y, X, beta)
{
  n <- dim(X)[1]
  beta <- matrix(beta, ncol = 1)
  y.mod <- 1*(y == 1) + 0*(y == -1)
  pi <- exp(X %*% beta) / (1 + exp(X%*%beta))  
  rtn <- colSums(X* as.numeric(y.mod - pi))
  return(n*rtn)
}


#################################################
## The following is a general function that
## implements the regular gradient ascent
## the stochastic gradient ascent and 
## mini-batch stochastic gradient ascent
#################################################
SGA <- function(y, X, batch.size = dim(X)[1], t = 0.1, t_cv = 0.1, max.iter = dim(X)[1], adapt = FALSE)
{  
  p <- dim(X)[2]
  n <- dim(X)[1]
  
  # create the mini-batches
  permutation <- sample(1:n, replace = FALSE)
  K <- floor(n/batch.size)
  batch.index <- split(permutation, rep(1:K, length = n, each = n/K))
  
  # index for choosing the mini-batch
  count <- 0
  
  beta_k = rep(0, p) # start at all 0s
  beta_k_cv <- beta_k # start at all 0s
  
  track.gradient <- matrix(0, nrow = max.iter, ncol = p)
  track.gradient[1,] <- (f_vanilla.gradient(y = y, X= X, beta = beta_k)/n)
  
  track.cv.gradient <- matrix(0, nrow = max.iter, ncol = p)
  track.cv.gradient[1,] <- cv.gradient(y = y, X= X, beta = beta_k_cv)
  
  track.beta <- matrix(0, nrow = max.iter, ncol = p)
  track.beta_cv <- matrix(0, nrow = max.iter, ncol = p)
  
  # saving the running mean of the estimates of theta^*
  mean_beta <- rep(0,p)
  mean_beta_cv <- rep(0,p)
  
  # tk: in case we want t_k
  tk <- t
  
  for(iter in 1:max.iter)
  {
    count <- count+1
    
    if((count-1) %% K == 0) count <- 1  # when all batches finish, restart the batches
    
    # Batch of data
    y.batch <- y[batch.index[[count]] ]
    X.batch <- matrix(X[batch.index[[count]], ], nrow = batch.size)
    
    # SGA step
    beta_k = beta_k + tk* f_vanilla.gradient(y = y.batch, X = X.batch, beta = beta_k)/batch.size
    est <- beta_k
    track.gradient[iter,] <- f_vanilla.gradient(y = y, X = X, beta = est)/n
    track.beta[iter, ] <- est
  }
  
  tk <- t_cv  
  count <- 0
  for(iter_cv in 1:max.iter)  
  {
    count <- count+1

    if ((count-1) %% K == 0) count <- 1  # when all batches finish, restart the batches
    
    # Batch of data
    y.batch <- y[batch.index[[count]] ]
    X.batch <- matrix(X[batch.index[[count]], ], nrow = batch.size)
    
    # SG_CV step
    beta_k_cv = beta_k_cv + tk*cv.gradient(y = y.batch, X= X.batch, beta = beta_k_cv)
    est_cv <- beta_k_cv
    track.cv.gradient[iter_cv,] <- cv.gradient(y = y, X= X, beta = est_cv)
    track.beta_cv[iter_cv, ] <- est_cv
  }
  
  rtn <- list("iter" = iter, "est" = est, "grad" = track.gradient[1:iter,],
              "iter_cv" = iter_cv, "est_cv" = est_cv, "grad_cv" = track.cv.gradient[1:iter_cv,], 
              "iterates" = track.beta[1:iter, ], "iterates_cv" = track.beta_cv[1:iter_cv, ])
  return(rtn)
}

#Generating Data
set.seed(10)
p <- 5
n <- 1e4
X <- matrix(rnorm(n*(p)), nrow = n, ncol = p)

beta = rnorm(n=length(X[1,]))

p <- exp(X %*% beta)/(1 + exp(X%*%beta))
y <- rbinom(n, size = 1, prob = p)
y.actual = y
y[which(y==0)] = -1

# Useful values for computing class-specific functions.
X_pos <- X[which(y==1),]
X_neg <- X[which(y==-1),]

n_pos <- length(X_pos[,1])
n_neg <- length(X_neg[,1])

X_pos_mean <- colMeans(X_pos)
X_pos_var <- var(X_pos)
X_neg_mean <- colMeans(X_neg)
X_neg_var <- var(X_neg)

# True MLE
true.mle = glm(y.actual ~ X-1, family = "binomial")
coeff <- true.mle$coeff

# Tuned value of t and t_cv
ga <- SGA(y, X, batch.size = 1e4, t = .001, t_cv = 0.1, max.iter = 1e4)
b100 <- SGA(y, X, batch.size = 100, t = .001, t_cv = 0.1, max.iter = 1e4) 

# Log-likelihood for the standard and CV iterates.
likelihood = apply(b100$iterates, 1, log.logit)
likelihood_cv = apply(b100$iterates_cv, 1, log.logit)

# Output Analysis
grDevices::pdf("./Plots/neurips_likelihood_zoomed.pdf", width = 6, height = 6)
par(mfrow = c(1, 1))
index <- 1:(1e4-299)
plot(index, likelihood[300:1e4], type="l", col = "blue", xlab = "Steps", ylab = "Log Likelihood")
lines(index, likelihood_cv[300:1e4], col = "red")
true.ml = log.logit(coeff)
abline(h = true.ml, col = "green")
legend("bottomright", legend = c("Standard-100", "VarRed-100", "True MLE"),
       col = c("blue", "red", "green"), lty = c(1, 1, 1))
dev.off()