% MATLAB ode45 = Dormand-Prince 5(4)
% Same tolerances as all other frameworks

sizes = [500, 1000, 2000, 4000, 8000];
tau = 0.01;
t_span = [0, 1];
n_warmup = 1;
n_runs = 5;
rtol = 1e-4;
atol = 1e-6;

fprintf('\nMATLAB ode45 (Dormand-Prince) Benchmark\n');
fprintf('%s\n', repmat('=', 1, 50));

results = struct();
for idx = 1:length(sizes)
    N = sizes(idx);

    rng(42);
    W = single(randn(N, N)) * single(1.5 / sqrt(N));
    b = single(randn(N, 1));
    y0 = single(randn(N, 1)) * 0.1;

    rhs = @(t, y) (-y + tanh(W * y + b)) / tau;
    opts = odeset('RelTol', rtol, 'AbsTol', atol);

    for w = 1:n_warmup
        [~, ~] = ode45(rhs, t_span, double(y0), opts);
    end

    times = zeros(n_runs, 1);
    for r = 1:n_runs
        tic;
        [~, ~] = ode45(rhs, t_span, double(y0), opts);
        times(r) = toc;
    end
    med = median(times);
    std_val = std(times);
    fprintf('  N=%5d: %.4fs +/- %.4fs\n', N, med, std_val);

    results.(sprintf('N%d', N)).median = med;
    results.(sprintf('N%d', N)).std = std_val;
end

% Save as JSON-like text
fid = fopen(fullfile(fileparts(mfilename('fullpath')), 'results_dp5_matlab.txt'), 'w');
for idx = 1:length(sizes)
    N = sizes(idx);
    r = results.(sprintf('N%d', N));
    fprintf(fid, '%d %.6f %.6f\n', N, r.median, r.std);
end
fclose(fid);
fprintf('\nResults saved to results_dp5_matlab.txt\n');
