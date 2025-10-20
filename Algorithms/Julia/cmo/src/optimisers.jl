# src/optimisers.jl

import .core
using LinearAlgebra: norm
using ForwardDiff: gradient

export GradientDescent

mutable struct GradientDescent <: Optimiser
    α::Real
end

function OptimiserStep(
        opt::GradientDescent,
        func::Function,
        state::OptimiserState;
        grad::Union{Nothing, Function} = nothing
)::OptimiserState
    if grad !== nothing
        g = grad(state.x)
    else
        g = gradient(func, state.x)
    end
    x_new = state.x .- opt.α .* g
    converged = norm(g) < 1e-6
    return OptimiserState(
        x_new,
        func(x_new),
        state.k + 1,
        converged
    )
end
