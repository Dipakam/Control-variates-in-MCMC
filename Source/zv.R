# Implement Algorithm 1 (Post-processing ZV) of the Report
postprocess.zv <- function(theta, gradient, g){
  z <- t(gradient)/2
  var_z = var(z)
  g_theta = g(t(theta))
  cov_gz = cov(g_theta, z)
  a = solve(var_z) %*% cov_gz
  return ( t(g_theta) + t(a) %*% t(z))
}