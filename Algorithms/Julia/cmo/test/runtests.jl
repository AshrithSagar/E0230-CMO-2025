# test/runtests.jl

using cmo
using Test: @testset, @test
using LinearAlgebra: Symmetric, I
using Random: seed!

@testset "Rosenbrock tests" begin
    f(𝐱) = cmo.Functions.rosenbrock(𝐱)
    @test f([1, 1]) == 0
    @test f([0, 0]) == 1
end

@testset "Gradient descent tests" begin
    f(𝐱) = cmo.Functions.rosenbrock(𝐱)
    opt = cmo.GradientDescent(1e-3)
    𝐱₀ = [1, 2, 1]
    𝐱ˢᵗᵃʳ = cmo.optimise(opt, f, 𝐱₀)
    𝐱ᵒᵖᵗ = [1, 1, 1]
    @test 𝐱ˢᵗᵃʳ≈𝐱ᵒᵖᵗ atol=1e-2
end

@testset "Conjugate gradient tests" begin
    seed!(25)
    Q = Symmetric(randn(3, 3)' * randn(3, 3) + 3I)
    h = randn(3)
    f(𝐱) = cmo.Functions.convex_quadratic(𝐱, Q = Q, h = h)
    opt = cmo.ConjugateGradient()
    𝐱₀ = [1, 2, 1]
    𝐱ˢᵗᵃʳ = cmo.optimise(opt, f, 𝐱₀)
    𝐱ᵒᵖᵗ = -Q \ h
    @test 𝐱ˢᵗᵃʳ≈𝐱ᵒᵖᵗ atol=1e-2
end
