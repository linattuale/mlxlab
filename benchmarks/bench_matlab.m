% MATLAB ODE benchmark: ode45 and hand-rolled RK4 on rate network
% Run: matlab -batch "run('bench_matlab.m')"

sizes = [100, 500, 1000, 2000];
dt = 0.001;
tau = 0.01;
t_span = [0, 1];
n_warmup = 1;
n_runs = 5;

fprintf('\nMATLAB ODE Benchmark (CPU)\n');
fprintf('%s\n', repmat('=', 1, 60));

for idx = 1:length(sizes)
    N = sizes(idx);
    fprintf('\n  N = %d\n', N);

    rng(42);
    W = single(randn(N, N)) * single(0.5 / sqrt(N));
    b = single(randn(N, 1));
    y0 = single(randn(N, 1)) * 0.1;

    % --- ode45 (adaptive) ---
    rhs_ode45 = @(t, y) (-y + tanh(W * y + b)) / tau;
    opts = odeset('MaxStep', dt, 'RelTol', 1e-4, 'AbsTol', 1e-6);

    for w = 1:n_warmup
        [~, ~] = ode45(rhs_ode45, t_span, double(y0), opts);
    end

    times_ode45 = zeros(n_runs, 1);
    for r = 1:n_runs
        tic;
        [~, ~] = ode45(rhs_ode45, t_span, double(y0), opts);
        times_ode45(r) = toc;
    end
    med_ode45 = median(times_ode45);
    std_ode45 = std(times_ode45);
    fprintf('    ode45 (adaptive): %.4fs +/- %.4fs\n', med_ode45, std_ode45);

    % --- Hand-rolled RK4 (fixed step, float32) ---
    times_rk4 = zeros(n_runs, 1);
    for r = 1:n_runs
        tic;
        t = t_span(1);
        y = y0;
        while t < t_span(2)
            h = min(dt, t_span(2) - t);
            k1 = (-y + tanh(W * y + b)) / tau;
            k2 = (-(y + h/2*k1) + tanh(W * (y + h/2*k1) + b)) / tau;
            k3 = (-(y + h/2*k2) + tanh(W * (y + h/2*k2) + b)) / tau;
            k4 = (-(y + h*k3) + tanh(W * (y + h*k3) + b)) / tau;
            y = y + (h/6) * (k1 + 2*k2 + 2*k3 + k4);
            t = t + h;
        end
        times_rk4(r) = toc;
    end
    med_rk4 = median(times_rk4);
    std_rk4 = std(times_rk4);
    fprintf('    RK4   (fixed dt): %.4fs +/- %.4fs\n', med_rk4, std_rk4);
end

fprintf('\nDone.\n');
