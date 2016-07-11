function [m, S, z] = gm(a,A,b,B)

% multiplies two Gaussians together, whose means are given by a, b, and
% whose covariances are given by A, B.
%
% (C) Copyright Marc Deisenroth, 2008-09-22

D = size(A,1); % dimension

L = chol(A+B)';                        % cholesky factorization of the covariance
alpha = L\(a-b);

S = A*((A+B)\B); % covariance matrix
m = B*((A+B)\a) + A*((A+B)\b); % mean
z = exp(-0.5*sum(alpha.^2,1))./((2*pi)^(0.5*D)*prod(diag(L)));