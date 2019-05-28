using LinearAlgebra
function kmeans(x, k; maxiters=100, tol=1e-5)
    N = length(x)
    n = length(x[1])
    distances = zeros(N) # 用来存储每个点距离最近中心点的距离

    reps = [zeros(n) for j=1:k] # 初始化k个中心点

    assignment = [rand(1:k) for i in 1:N] # 随机给每个向量分一个类

    Jprevious = Inf
    for iter = 1:maxiters

        for j = 1:k 
            group = [i for i=1:N if assignment[i] == j]
            reps[j] = sum(x[group]) / length(group)
        end
        for i in 1:N
            (distances[i], assignment[i]) =
                    findmin([norm(x[i] - reps[j]) for j in 1:k]);
        end
        J = norm(distances)^2 / N
        println("Iteration ", iter, ": Jclust = ", J, ".")
        if iter > 1 && abs(J - Jprevious) < tol * J
            return assignment, reps
        end
        Jprevious = J
    end
end

X = vcat([0.3*randn(2) for i=1:100],
    [[1, 1] + 0.3*randn(2) for i=1:100],
    [[1, -1] + 0.3*randn(2) for i=1:100]
)

assignment, reps = kmeans(X, 3)
