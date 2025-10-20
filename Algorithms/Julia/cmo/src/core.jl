# src/core.jl

abstract type Optimiser end

mutable struct OptimiserState
    x::AbstractVector{<:Real}
    f::Real
    k::Unsigned
    converged::Bool
end

function OptimiserStart(opt::Optimiser, func::Function, x0::AbstractVector{<:Real})
    return OptimiserState(x0, func(x0), 0, false)
end

function OptimiserStep(
        opt::Optimiser, func::Function, state::OptimiserState
)::OptimiserState
    error("OptimiserStep not implemented for $(typeof(opt))")
end

function optimise(opt::Optimiser, func::Function, x0::AbstractVector{<:Real})
    state = OptimiserStart(opt, func, x0)
    while !state.converged
        state = OptimiserStep(opt, func, state)
    end
    return state.x
end
