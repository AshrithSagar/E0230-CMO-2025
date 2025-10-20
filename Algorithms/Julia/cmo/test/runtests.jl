# test/runtests.jl

using Test
using cmo

@testset "Rosenbrock tests" begin
    f(x) = cmo.Functions.rosenbrock(x)

    @test f([1, 1]) == 0
    @test f([0, 0]) == 1
end

@testset "Gradient descent tests" begin
    f(x) = cmo.Functions.rosenbrock(x)
    opt = cmo.GradientDescent(1e-3)
    x0 = [1, 2, 1]
    xstar = cmo.optimise(opt, f, x0)
    @test xstar≈[1, 1, 1] atol=1e-2
end
