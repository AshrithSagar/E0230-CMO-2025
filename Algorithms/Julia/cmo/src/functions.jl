# src/functions.jl
module Functions

using LinearAlgebra: dot, Symmetric

function convex_quadratic(
        𝐱::AbstractVector{<:Real};
        Q::Symmetric{<:Real} = Symmetric(I, length(𝐱)),
        h::AbstractVector{<:Real} = zeros(eltype(𝐱), length(𝐱)),
        c::Real = 0
)
    return 0.5 * dot(𝐱, Q, 𝐱) + dot(h, 𝐱) + c
end

# =========================================

function rosenbrock(
        𝐱::AbstractVector{<:Real};
        a::Real = 1,
        b::Real = 100
)
    return sum(b * (𝐱[2:end] .- 𝐱[1:(end - 1)] .^ 2) .^ 2 + (a .- 𝐱[1:(end - 1)]) .^ 2)
end

function rosenbrock_grad(
        𝐱::AbstractVector{<:Real};
        a::Real = 1,
        b::Real = 100
)
    dim = length(𝐱)
    grad = zeros(eltype(𝐱), dim)
    for i in 1:dim
        if i > 1
            grad[i] += 2b * (𝐱[i] - 𝐱[i - 1]^2)
        end
        if i < dim
            grad[i] += -4b * (𝐱[i + 1] - 𝐱[i]^2) * 𝐱[i] - 2 * (a - 𝐱[i])
        end
    end
    return grad
end

end  # module
