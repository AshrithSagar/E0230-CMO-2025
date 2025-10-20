# src/optimisers.jl

import .core
using LinearAlgebra: norm
using ForwardDiff: gradient

export GradientDescent

mutable struct GradientDescent <: Optimiser
    α::Real
end

function OptimiserStep(
        opt::GradientDescent, func::Function, state::OptimiserState
)::OptimiserState
    x_new = state.x .- opt.α .* gradient(func, state.x)
    converged = norm(x_new .- state.x) < 1e-12
    return OptimiserState(
        x_new,
        func(x_new),
        state.k + 1,
        converged
    )
end
