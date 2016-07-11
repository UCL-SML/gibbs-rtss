function x = gaussian(m, S, n)

% compute n samples from a multivariate Gaussian with mean m and covariance
% S
% 
% input parameters:
% m
% S
% n
%
% 2010-07-07

if nargin < 3, n = 1; end

x = bsxfun(@plus, m(:), chol(S)'*randn(length(m),n));
