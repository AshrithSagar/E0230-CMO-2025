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

end  # module
