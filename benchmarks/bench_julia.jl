"""
Julia ODE benchmark: Tsit5 (CPU) on rate network dy/dt = (-y + tanh(W*y + b)) / tau
Run: julia bench_julia.jl
"""

using OrdinaryDiffEq
using LinearAlgebra
using Random
using Printf
using Statistics

const SIZES = [100, 500, 1000, 2000]
const T_SPAN = (0.0, 1.0)
const DT = 0.001
const TAU = 0.01f0
const N_WARMUP = 1
const N_RUNS = 5

function make_system(N; seed=42)
    rng = MersenneTwister(seed)
    W = Float32.(randn(rng, N, N)) .* Float32(0.5 / sqrt(N))
    b = Float32.(randn(rng, N))
    y0 = Float32.(randn(rng, N)) .* 0.1f0
    return W, b, y0
end

function bench_tsit5(N)
    W, b, y0 = make_system(N)

    function rhs!(dy, y, p, t)
        mul!(dy, W, y)
        @. dy = (-y + tanh(dy + b)) / TAU
        return nothing
    end

    prob = ODEProblem(rhs!, y0, T_SPAN)

    # Warmup
    for _ in 1:N_WARMUP
        solve(prob, Tsit5(); dt=DT, adaptive=false, save_everystep=false)
    end

    # Timed runs
    times = Float64[]
    for _ in 1:N_RUNS
        t0 = time_ns()
        solve(prob, Tsit5(); dt=DT, adaptive=false, save_everystep=false)
        push!(times, (time_ns() - t0) / 1e9)
    end

    med = sort(times)[div(length(times), 2) + 1]
    std_val = std(times)
    return med, std_val
end

function bench_rk4(N)
    W, b, y0 = make_system(N)

    function rhs!(dy, y, p, t)
        mul!(dy, W, y)
        @. dy = (-y + tanh(dy + b)) / TAU
        return nothing
    end

    prob = ODEProblem(rhs!, y0, T_SPAN)

    # Warmup
    for _ in 1:N_WARMUP
        solve(prob, RK4(); dt=DT, adaptive=false, save_everystep=false)
    end

    # Timed runs
    times = Float64[]
    for _ in 1:N_RUNS
        t0 = time_ns()
        solve(prob, RK4(); dt=DT, adaptive=false, save_everystep=false)
        push!(times, (time_ns() - t0) / 1e9)
    end

    med = sort(times)[div(length(times), 2) + 1]
    std_val = std(times)
    return med, std_val
end

function main()
    println("\nJulia ODE Benchmark (CPU, Apple Accelerate BLAS)")
    println("=" ^ 60)

    for N in SIZES
        println("\n  N = $N")
        med_tsit5, std_tsit5 = bench_tsit5(N)
        @printf("    Tsit5 (fixed dt): %.4fs +/- %.4fs\n", med_tsit5, std_tsit5)

        med_rk4, std_rk4 = bench_rk4(N)
        @printf("    RK4   (fixed dt): %.4fs +/- %.4fs\n", med_rk4, std_rk4)
    end
end

main()
