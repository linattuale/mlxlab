using OrdinaryDiffEq, LinearAlgebra, Random, Printf, Statistics, JSON3

const SIZES = [500, 1000, 2000, 4000, 8000]
const T_SPAN = (0.0, 1.0)
const TAU = 0.01f0
const RTOL = 1e-4
const ATOL = 1e-6
const N_WARMUP = 1
const N_RUNS = 5

function make_system(N; seed=42)
    rng = MersenneTwister(seed)
    W = Float32.(randn(rng, N, N)) .* Float32(1.5 / sqrt(N))
    b = Float32.(randn(rng, N))
    y0 = Float32.(randn(rng, N)) .* 0.1f0
    return W, b, y0
end

function bench_dp5(N)
    W, b, y0 = make_system(N)

    function rhs!(dy, y, p, t)
        mul!(dy, W, y)
        @. dy = (-y + tanh(dy + b)) / TAU
        return nothing
    end

    prob = ODEProblem(rhs!, y0, T_SPAN)

    for _ in 1:N_WARMUP
        solve(prob, DP5(); reltol=RTOL, abstol=ATOL, save_everystep=false)
    end

    times = Float64[]
    for _ in 1:N_RUNS
        t0 = time_ns()
        solve(prob, DP5(); reltol=RTOL, abstol=ATOL, save_everystep=false)
        push!(times, (time_ns() - t0) / 1e9)
    end

    med = median(times)
    std_val = std(times)
    return med, std_val
end

function main()
    println("\nJulia DP5 (Dormand-Prince) Benchmark")
    println("=" ^ 50)

    results = Dict{String, Any}()
    for N in SIZES
        med, std_val = bench_dp5(N)
        @printf("  N=%5d: %.4fs +/- %.4fs\n", N, med, std_val)
        results[string(N)] = Dict("median" => med, "std" => std_val)
    end

    # Save results for merge into Python results file
    open(joinpath(@__DIR__, "results_dp5_julia.json"), "w") do f
        JSON3.write(f, results)
    end
    println("\nResults saved to results_dp5_julia.json")
end

main()
