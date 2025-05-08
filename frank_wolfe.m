% read in the training matrix
M_train = readmatrix('M_train.csv');
[users, movies] = size(M_train);

% define f and nabla_f
function f_val = f(X, M)
    square_diff = (X - M).^2;
    M(M ~= 0) = 1;
    f_val = sum(sum(M .* square_diff));
end
function nabla_f_val = nabla_f(X, M)
    diff = 2 * (X - M);
    M(M ~= 0) = 1;
    nabla_f_val = M .* diff;
end

% execute conditional gradient method
X = zeros(users, movies);
T = 2000;
theta = 600;
for t = 1:T
    Y = nabla_f(X, M_train);
    [U,S,V] = svds(Y, 1);
    v = -theta * U * V.';
    tau = 2 / (t + 2);
    X = (1 - tau) * X + tau * v;
end

% evaluate test error
M_test = readmatrix('M_test.csv');
disp(['Using the Frank-Wolfe method, we achieve test error ', num2str(f(X, M_test)), '.'])