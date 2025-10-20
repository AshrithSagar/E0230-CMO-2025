# src/core.jl

abstract type Optimiser end
abstract type OptimiserState end

function OptimiserStart(
        opt::Optimiser,
        func::Function,
        𝐱₀::AbstractVector{<:Real};
        grad::Union{Nothing, Function} = nothing,
        hess::Union{Nothing, Function} = nothing
)
    error("OptimiserStart not implemented for $(typeof(opt))")
end

function OptimiserStep(
        opt::Optimiser,
        func::Function,
        state::OptimiserState;
        grad::Union{Nothing, Function} = nothing,
        hess::Union{Nothing, Function} = nothing
)::OptimiserState
    error("OptimiserStep not implemented for $(typeof(opt))")
end

function optimise(
        opt::Optimiser,
        func::Function,
        𝐱₀::AbstractVector{<:Real};
        grad::Union{Nothing, Function} = nothing,
        hess::Union{Nothing, Function} = nothing
)
    state = OptimiserStart(opt, func, 𝐱₀; grad, hess)
    while !state.converged
        state = OptimiserStep(opt, func, state; grad, hess)
    end
    return state.𝐱
end
