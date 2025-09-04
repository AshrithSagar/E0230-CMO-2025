# oracle.jl
abstract type AbstractOracle end

abstract type ZerothOrderOracle <: AbstractOracle end
function eval(o::ZerothOrderOracle, x::Vector{Real})
    throw(MethodError("eval", (o, x)))
end

abstract type FirstOrderOracle <: ZerothOrderOracle end
function gradient(o::FirstOrderOracle, x::Vector{Real})
    throw(MethodError("gradient", (o, x)))
end

abstract type SecondOrderOracle <: FirstOrderOracle end
function hessian(o::SecondOrderOracle, x::Vector{Real})
    throw(MethodError("hessian", (o, x)))
end

struct QuadraticOracle <: SecondOrderOracle
    A::Matrix{Real}
    b::Vector{Real}
    c::Real
end
function eval(o::QuadraticOracle, x::Vector{Real})
    return 0.5 * x' * o.A * x + dot(o.b, x) + o.c
end
function gradient(o::QuadraticOracle, x::Vector{Real})
    return o.A * x + o.b
end
function hessian(o::QuadraticOracle, x::Vector{Real})
    return o.A
end
