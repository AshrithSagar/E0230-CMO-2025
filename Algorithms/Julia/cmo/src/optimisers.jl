# src/optimisers.jl

import .core
using LinearAlgebra: norm, dot, Symmetric
using ForwardDiff: gradient, hessian

export GradientDescent, ConjugateGradient

# ===== Gradient Descent Optimiser =====
mutable struct GradientDescent <: Optimiser
    α::Real
end

mutable struct GradientDescentState <: OptimiserState
    𝐱::AbstractVector{<:Real}
    f::Real
    k::Unsigned
    converged::Bool
end

function OptimiserStart(
        opt::GradientDescent,
        func::Function,
        𝐱₀::AbstractVector{<:Real};
        grad::Union{Nothing, Function} = nothing,
        hess::Union{Nothing, Function} = nothing
)::GradientDescentState
    if grad !== nothing
        ∇f = grad(𝐱₀)
    else
        ∇f = gradient(func, 𝐱₀)
    end

    return GradientDescentState(
        𝐱₀, func(𝐱₀), 0, false
    )
end

function OptimiserStep(
        opt::GradientDescent,
        func::Function,
        state::GradientDescentState;
        grad::Union{Nothing, Function} = nothing,
        hess::Union{Nothing, Function} = nothing
)::GradientDescentState
    𝐱ₖ = state.𝐱
    α = opt.α

    if grad !== nothing
        ∇f = grad(𝐱ₖ)
    else
        ∇f = gradient(func, 𝐱ₖ)
    end

    𝐱ₖ₊₁ = 𝐱ₖ - α * ∇f

    converged = norm(∇f) < 1e-6

    return GradientDescentState(
        𝐱ₖ₊₁, func(𝐱ₖ₊₁), state.k + 1, converged
    )
end

# ===== Conjugate Gradient Optimiser =====
mutable struct ConjugateGradient <: Optimiser end

mutable struct ConjugateGradientState <: OptimiserState
    𝐱::AbstractVector{<:Real}
    f::Real
    k::Unsigned
    converged::Bool

    𝐐::Symmetric{<:Real}
    𝐫::AbstractVector{<:Real}
    𝐩::AbstractVector{<:Real}
end

function OptimiserStart(
        opt::ConjugateGradient,
        func::Function,
        𝐱₀::AbstractVector{<:Real};
        grad::Union{Nothing, Function} = nothing,
        hess::Union{Nothing, Function} = nothing
)::ConjugateGradientState
    if grad !== nothing
        ∇f = grad(𝐱₀)
    else
        ∇f = gradient(func, 𝐱₀)
    end

    if hess !== nothing
        ∇²f = hess(𝐱₀)
    else
        ∇²f = hessian(func, 𝐱₀)
    end

    𝐐 = Symmetric(∇²f)
    𝐫₀ = ∇f
    𝐩₀ = -∇f

    return ConjugateGradientState(
        𝐱₀, func(𝐱₀), 0, false, 𝐐, 𝐫₀, 𝐩₀
    )
end

function OptimiserStep(
        opt::ConjugateGradient,
        func::Function,
        state::ConjugateGradientState;
        grad::Union{Nothing, Function} = nothing,
        hess::Union{Nothing, Function} = nothing
)::ConjugateGradientState
    𝐱ₖ = state.𝐱
    𝐐 = state.𝐐
    𝐫ₖ = state.𝐫
    𝐩ₖ = state.𝐩

    αₖ = dot(𝐫ₖ, 𝐫ₖ) / dot(𝐩ₖ, 𝐐, 𝐩ₖ)
    𝐱ₖ₊₁ = 𝐱ₖ + αₖ * 𝐩ₖ
    𝐫ₖ₊₁ = 𝐫ₖ + αₖ * 𝐐 * 𝐩ₖ
    βₖ₊₁ = dot(𝐫ₖ₊₁, 𝐫ₖ₊₁) / dot(𝐫ₖ, 𝐫ₖ)
    𝐩ₖ₊₁ = -𝐫ₖ₊₁ + βₖ₊₁ * 𝐩ₖ

    converged = norm(𝐫ₖ₊₁) < 1e-6

    return ConjugateGradientState(
        𝐱ₖ₊₁, func(𝐱ₖ₊₁), state.k + 1, converged, 𝐐, 𝐫ₖ₊₁, 𝐩ₖ₊₁
    )
end
