function [m, S, mxPred, SxPred, mzPred, SzPred, J, e1, e2] = ...
  gibbs_smoother(m, S, fhandle, Q, nX, NX, biX, ghandle, R, nZ, NZ, biZ, ...
  meas, f_params, g_params)
%
% Gibbs-sampling based algorithm for filtering and pre-computation of
% necessary parameters for smoothing during the forward sweep of a
% forward-backward-type algorithm
%
% inputs:
% m:                state mean
% S:                state covariance
% fhandle:          function handle to transition dynamics
% Q:                system noise covariance matrix
% nX:               number of samples to be drawn from p(x_{t-1}|z_{1:t-1})
% NX:               number of Gibbs iterations
% biX:              burn-in-period
% ghandle:          function handle to measurement function
% R:                measurement noise covariance
% nZ:               number of samples to be drawn from p(x_t|z_{1:t-1})
% NZ:               number of Gibbs iterations
% biZ:              burn-in period
% meas:             measured state
% fparams:          additional parameters to be passed on to the transition
%                   dynamics. Needs to be a cell array
% gparams:          additional parameters to be passed on to the
%                   measurement function. Needs to be a cell array
%
%
% returns:
% m:               filtered mean of latent state: E[x_t|z_{1:t}]
% S:               filtered covariance of latent state: cov[x_t|z_{1:t}]
% mxPred:          mean of time update: E[x_t|z_{1:t-1}]
% SxPred:          covariance of time update: cov[x_t|z_{1:t-1}]
% mzPred:          mean of measurement distriution p(z_t|z_{1:t-1})
% SzPred:          covariance of measurement distributin p(z_t|z_{1:t-1})
% J:               pre-computation for smoothing
% e1, e1:          discrepancies between theoretically identical marginal
%                  distributions
%
% (C) Copyright by Marc Deisenroth, 
% Last modified: 2016-07-11
%
% References:
%
% Marc Peter Deisenroth and Henrik Ohlsson
% A General Perspective on Gaussian Filtering and Smoothing:
% Explaining Current and Deriving New Algorithms
% in Proceedings of the 2011 American Control Conference (ACC 2011)


D = size(S,1);  % dimension of input
E = size(Q,1);  % dimension of state
F = size(R,1);  % dimension of measurement


% 1) time update
% compute p(x_{t-1}, x_t)
[mu_jointX Sigma_jointX samples] = ...
  gibbs_joint(m, S, fhandle, Q, nX, NX, biX, [], f_params);

% mxOld = mu_jointX(1:D);
mxPred = mu_jointX(D+1:2*D);

% Sxx = Sigma_jointX(1:D,1:D);
C = Sigma_jointX(1:D,D+1:2*D);
SxPred = Sigma_jointX(D+1:2*D,D+1:2*D);

% 2) measurement update
% compute p(x_t, z_t)
[mu_jointZ Sigma_jointZ] = ...
  gibbs_joint(mxPred, SxPred, ghandle, R, nZ, NZ, biZ, samples(D+1:end,:), g_params);

mx = mu_jointZ(1:D);
mzPred = mu_jointZ(E+1:E+F);

Sxx = Sigma_jointZ(1:D,1:D);
Sxz = Sigma_jointZ(1:E,E+1:E+F);
SzPred = Sigma_jointZ(E+1:E+F, E+1:E+F);

% 3) filter distribution
L = chol(SzPred)';
B = L \ (Sxz');
m = mx + Sxz * (SzPred \ (meas - mzPred));
S = Sxx - B' * B;

% 4) pre-computation for smoothing
J = C / SxPred;

% some errors that should converge to zero when sampling from the
% stationary distribution
% alternative way: infer the parameters of the joint p(x_{t-1}, x_t, z_t)
% at once
e1 = norm(mxPred - mx) / norm(mxPred + mx);
e2 = norm(Sxx - SxPred) / norm(Sxx + SxPred);

% disp(num2str([e1 e2]));



%%
function [mu_joint Sigma_joint joint_samples] = ...
  gibbs_joint(muMarg, SigmaMarg, fhandle, Q, n, N, bi, x, params)

% Given the mean and the covariance of p(x), compute the mean and the
% covariance of the joint p(x,h(x)) for a function h, with prior
% distributions
% p(mu_joint) = N(m,S)
% p(Sigma_joint) = IW(Psi, kappa)
%
% inputs:
% muMarg      mean of p(x)
% SigmaMarg   covariance of p(x)
% fhandle     function handle for h
% Q           noise covariance matrix; could be inferred from data if
%             unknown
% n           number of samples to be generated from p(x)
% N           number of Gibbs iterations
% bi          burn-in period
% x           data set for marginal distribution
% params      additional set of parameters required to evaluate h
%
% 
% (C) Copyright by Marc Deisenroth
% last modified: 2011-02-17
%
% 
if ~exist('n','var'); n = 200; end
if ~exist('N','var'); N = 1000; end
if ~exist('x', 'var') || isempty(x) % generate data set from known marginal
  x = gaussian(muMarg, SigmaMarg, n);   % samples from marginal prior
end
if nargin(fhandle) > 1
  z = fhandle(x,params{:}); % you could have control signals in here
else
  z = fhandle(x);
end

if ~exist('Q', 'var') || isempty(Q)
  x = [x; z];
else
  x = [x; z + chol(Q)'*randn(size(z))]; % samples from the joint
end

D = size(muMarg,1);
E = size(z,1);

% initialization
Sigma = cell(1,N);
Sigma_joint = zeros(D+E);

% hyper-parameters for mean of joint
m = zeros(D+E,1); m(1:D) = muMarg; % use what we know about mean
S = eye(D+E);
mu(:,1) = gaussian(m, S); % sample initial mean

% hyper-parameters for covariance of joint
Psi = eye(D+E); Psi(1:D,1:D) = SigmaMarg; % use what we know about covariance
Psi = (Psi + Psi')/2;
kappa = 2 + (D+E);
Sigma{1} = iwishrnd(Psi, kappa); % sample initial covariance

d = zeros(1,N);
for i = 1:N
  fprintf('%s %d\r', 'Gibbs iteration #', i);

  % posterior hyper-parameters for mean of joint distribution
  [m, S] = gm(mean(x,2), Sigma{i}/n, m, S);

  % sample mu
  mu(:,i+1) = gaussian(m, S);

  % posterior hyper-parameters for covariance of joint distribution
  kappa = kappa + n; 
  x2 = bsxfun(@minus, x, mu(:,i+1));
  Psi = Psi + x2*x2'; Psi = (Psi + Psi')/2;

  % sample Sigma
  Sigma{i+1} = iwishrnd(Psi, kappa);

  d(i+1) = det(Sigma{i+1}); % only for convergence analysis
end

% (unbiased) estimate of mean/covariance of joint distribution
mu_joint = mean(mu(:,bi+1:end),2);
for j = bi+1:N; Sigma_joint = Sigma_joint + Sigma{j}/(N-bi); end

joint_samples = x; % return the samples. might be useful in subsequent step

fprintf('\n');