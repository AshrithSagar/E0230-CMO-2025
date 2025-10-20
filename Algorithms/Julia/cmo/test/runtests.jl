# test/runtests.jl

using Test
using cmo.Functions

@testset "Rosenbrock tests" begin
    f(x) = Functions.rosenbrock(x)

    @test f([1.0, 1.0]) == 0.0
    @test f([0.0, 0.0]) == 1.0
end
