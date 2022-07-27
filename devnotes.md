cost

eps = 1.e-12
linp = dot(data, params)
pi = 1 / (1 + exp(-linp))
if (abs((target_variable) - 1) < eps) {
    logL += log(pi)
} else {
    logL += log(1.-pi)
}

gradient

mvaxpy is just vecsum(vecmul(data * fact))

quasinewton bfgs
