N=5000; tau=0.01; dt=0.001; rng(42);
W=single(randn(N,N))*single(0.5/sqrt(N));
b=single(randn(N,1));
y0=single(randn(N,1))*0.1;

% warmup
t=0; y=y0;
while t<1
    h=min(dt,1-t);
    k1=(-y+tanh(W*y+b))/tau;
    k2=(-(y+h/2*k1)+tanh(W*(y+h/2*k1)+b))/tau;
    k3=(-(y+h/2*k2)+tanh(W*(y+h/2*k2)+b))/tau;
    k4=(-(y+h*k3)+tanh(W*(y+h*k3)+b))/tau;
    y=y+(h/6)*(k1+2*k2+2*k3+k4);
    t=t+h;
end

times=zeros(3,1);
for r=1:3
    tic;
    t=0; y=y0;
    while t<1
        h=min(dt,1-t);
        k1=(-y+tanh(W*y+b))/tau;
        k2=(-(y+h/2*k1)+tanh(W*(y+h/2*k1)+b))/tau;
        k3=(-(y+h/2*k2)+tanh(W*(y+h/2*k2)+b))/tau;
        k4=(-(y+h*k3)+tanh(W*(y+h*k3)+b))/tau;
        y=y+(h/6)*(k1+2*k2+2*k3+k4);
        t=t+h;
    end
    times(r)=toc;
end
fprintf('MATLAB RK4 N=5000: %.4fs +/- %.4fs\n', median(times), std(times));
