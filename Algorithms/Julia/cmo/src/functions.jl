# src/functions.jl
module Functions

using LinearAlgebra: dot, Symmetric

function convex_quadratic(
        x::AbstractVector{<:Real};
        Q::Symmetric{<:Real}, h::AbstractVector{<:Real}, c::Real
)
    return 0.5 * dot(x, Q * x) + dot(h, x) + c
end

# =========================================

function rosenbrock(
        x::AbstractVector{<:Real};
        a::Real = 1, b::Real = 100
)
    return sum(b * (x[2:end] .- x[1:(end - 1)] .^ 2) .^ 2 + (a .- x[1:(end - 1)]) .^ 2)
end

function rosenbrock_grad(
        x::AbstractVector{<:Real};
        a::Real = 1, b::Real = 100
)
    dim = length(x)
    grad = zeros(eltype(x), n)
    for i in 1:dim
        if i > 1
            grad[i] += 2b * (x[i] - x[i - 1]^2)
        end
        if i < dim
            grad[i] += -4b * (x[i + 1] - x[i]^2) * x[i] - 2 * (a - x[i])
        end
    end
    return grad
end

end  # module
