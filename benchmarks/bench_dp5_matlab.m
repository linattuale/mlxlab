% MATLAB ode45 = Dormand-Prince 5(4), b=0, g=1.5
% Save all steps, report step counts

sizes = [500, 1000, 2000, 4000, 8000, 16000, 32000];
tau = 0.01;
t_span = [0, 1];
n_warmup = 1;
n_runs = 5;
rtol = 1e-4;
atol = 1e-6;

fprintf('\nMATLAB ode45 (Dormand-Prince) — b=0, g=1.5, save all steps\n');
fprintf('%s\n', repmat('=', 1, 56));

fid = fopen(fullfile(fileparts(mfilename('fullpath')), 'results_dp5_matlab.txt'), 'w');

for idx = 1:length(sizes)
    N = sizes(idx);

    rng(42);
    W = single(randn(N, N)) * single(1.5 / sqrt(N));
    y0 = single(randn(N, 1)) * 0.1;

    rhs = @(t, y) (-y + tanh(W * y)) / tau;
    opts = odeset('RelTol', rtol, 'AbsTol', atol, 'Refine', 1);

    for w = 1:n_warmup
        [~, ~] = ode45(rhs, t_span, double(y0), opts);
    end

    % Get step count
    [t_sol, ~] = ode45(rhs, t_span, double(y0), opts);
    n_steps = length(t_sol) - 1;

    runs = min(n_runs, 3 + 2*(N <= 8000));  % 3 runs for large N
    times = zeros(runs, 1);
    for r = 1:runs
        tic;
        [~, ~] = ode45(rhs, t_span, double(y0), opts);
        times(r) = toc;
    end
    med = median(times);
    std_val = std(times);
    fprintf('  N=%5d: %.4fs +/- %.4fs  (%d steps)\n', N, med, std_val, n_steps);
    fprintf(fid, '%d %.6f %.6f %d\n', N, med, std_val, n_steps);
end

fclose(fid);
fprintf('\nResults saved to results_dp5_matlab.txt\n');
