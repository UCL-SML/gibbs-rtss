% example: smoothing with Gibbs sampling, EKS, CKS, URTSS
%
% setup of nonlinear time series taken from
%
% A Doucet, S Godsill, C Andrieu:
% "On sequential Monte Carlo sampling methods for Bayesian filtering"
% Statistics and Computing (2000) 10, pp. 197--208, eqs. (73)--(74)
%
% (C) Copyright 2010-2011 by Marc Deisenroth,
% Last modified: 2016-07-11
%
%
% References:
%
% Marc Peter Deisenroth and Henrik Ohlsson
% A General Perspective on Gaussian Filtering and Smoothing:
% Explaining Current and Deriving New Algorithms
% in Proceedings of the 2011 American Control Conference (ACC 2011)

clear all; close all;

addpath ./util/
addpath ./filters/

set(0,'defaultaxesfontsize',20);
set(0,'defaultaxesfontunits', 'points')
set(0,'defaulttextfontsize',22);
set(0,'defaulttextfontunits','points')
set(0,'defaultaxeslinewidth',1);
set(0,'defaultlinelinewidth',2);
set(0,'DefaultAxesLineStyleOrder','-|--|:|-.');


plotOn = true; if plotOn; randn('state',1407); end;
printFig = false;

%% set up dynamic system

% system function
system_fct = @(x,t) 0.5*x + 25.*x./(1+x.^2) + 8.*cos(1.2*t);
system_fct2 = @(x,u,n,t) system_fct(x,t) + n;

% measurement function
meas_fct = @(x) x.^2/20;
meas_fct2 = @(x,u,n,t) meas_fct(x) + n;

%% evaluation measures

% negative log-likelihood per point
nllfun = @(xt, x, C) (0.5*log(C) + 0.5*(x-xt).^2./C + 0.5.*log(2*pi));

% RMSE
rmsefun = @(xt, x) sqrt(mean((x-xt).^2));

%% set noise parameters
Q = 1; % covariance system noise
R = 10; % covariance measurement noise

D = size(Q,2); % dimension of latent state
E = size(R,2); % dimension of measurement
%% prior parameters
mu0 = 0;  % mean
S0 = 5;   % covariance

%% other parameters
T = 50; % number of time steps in horizon

%% Gibbs smoother parameters

% parameters to infer p(x_{t-1}, x_t)
nX = 1000;  % size of generated data set
NX = 200;   % number of Gibbs iterations
biX = NX/2; % burn-in period

% parameters to infer p(x_t, z_t)
nZ = 1000;
NZ = 200;
biZ = NZ/2;
%% generate time series

x(:,1) = chol(S0)'*randn(D,1) + mu0; % sample initial state from prior
z(:,1) = meas_fct(x(:,1)) + chol(R)'*randn(E,1);

for t = 1:T
  x(:,t+1) = system_fct(x(:,t),t) + chol(Q)'*randn(D,1); % latent state
  z(:,t+1) = meas_fct(x(:,t+1)) + chol(R)'*randn(E,1); % measurement
end


%% memory allocations

% Gibbs
mean_gibbs_filter = zeros(D,T+1);
cov_gibbs_filter = cell(1,T+1);
mxPred_gibbs = zeros(D,T+1);
SxPred_gibbs = cell(1,T+1);
mzPred_gibbs = zeros(E,T+1);
SzPred_gibbs = cell(1,T+1);
J_gibbs = cell(T,1);

% EKF
mean_ekf = zeros(D,T+1);
cov_ekf = cell(1,T+1);
mxPred_ekf = zeros(D,T+1);
SxPred_ekf = cell(1,T+1);
mzPred_ekf = zeros(E,T+1);
SzPred_ekf = cell(1,T+1);
J_ekf = cell(T,1);

% CKF
mean_ckf = zeros(D,T+1);
cov_ckf = cell(1,T+1);
mxPred_ckf = zeros(D,T+1);
SxPred_ckf = cell(1,T+1);
mzPred_ckf = zeros(E,T+1);
SzPred_ckf = cell(1,T+1);
J_ckf = cell(T,1);

% UKF
mean_ukf = zeros(D,T+1);
cov_ukf = cell(1,T+1);
mxPred_ukf = zeros(D,T+1);
SxPred_ukf = cell(1,T+1);
mzPred_ukf = zeros(E,T+1);
SzPred_ukf = cell(1,T+1);
J_ukf = cell(T,1);


% initialize means and covariances
mean_gibbs_smoother = zeros(D,T+1);
cov_gibbs_smoother = cell(1,T+1);

mean_eks = zeros(D,T+1);
cov_eks = cell(1,T+1);

mean_cks = zeros(D,T+1);
cov_cks = cell(1,T+1);

mean_uks = zeros(D,T+1);
cov_uks = cell(1,T+1);


%% filtering and smoothing (forward-backward framework)

% forward sweep
mean_gibbs_filter(:,1) = mu0;
cov_gibbs_filter{1} = S0;

mean_ekf(:,1) = mu0;
cov_ekf{1} = S0;

mean_ckf(:,1) = mu0;
cov_ckf{1} = S0;

mean_ukf(:,1) = mu0;
cov_ukf{1} = S0;

for t = 1:T
  fprintf('\n');
  disp(['time step ' num2str(t) ' of ' num2str(T)]);
  
  % Gibbs filter
  [mean_gibbs_filter(:,t+1), cov_gibbs_filter{t+1}, mxPred_gibbs(:,t+1),...
    SxPred_gibbs{t+1}, mzPred_gibbs(:,t+1), SzPred_gibbs{t+1}, J_gibbs{t}] = ...
    gibbs_smoother(mean_gibbs_filter(:,t), cov_gibbs_filter{t}, system_fct, ...
    Q, nX, NX, biX, meas_fct, R, nZ, NZ, biZ, z(:,t+1), {t}, {});
  
  % EKF
  [mean_ekf(:,t+1), cov_ekf{t+1}, mxPred_ekf(:,t+1),...
    SxPred_ekf{t+1}, mzPred_ekf(:,t+1), SzPred_ekf{t+1}, J_ekf{t}] = ...
    ekf(system_fct, mean_ekf(:,t), cov_ekf{t}, meas_fct, ...
    z(:,t+1), Q, R, t);
  
  % CKF
  [mean_ckf(:,t+1), cov_ckf{t+1}, mxPred_ckf(:,t+1),...
    SxPred_ckf{t+1}, mzPred_ckf(:,t+1), SzPred_ckf{t+1}, J_ckf{t}] = ...
    sckf(mean_ckf(:,t), cov_ckf{t}, system_fct, meas_fct, ...
    z(:,t+1), Q, R, t);
  
  % UKF
  [mean_ukf(:,t+1), cov_ukf{t+1}, mxPred_ukf(:,t+1),...
    SxPred_ukf{t+1}, mzPred_ukf(:,t+1), SzPred_ukf{t+1}, J_ukf{t}] = ...
    ukf_add(mean_ukf(:,t), cov_ukf{t}, [], Q, system_fct2, z(:,t+1), R, meas_fct2, ...
    t, 1, 0, 2);
end

% backward sweep
mean_gibbs_smoother(:,T+1) = mean_gibbs_filter(:,T+1);
cov_gibbs_smoother{T+1} = cov_gibbs_filter{T+1};

mean_eks(:,T+1) = mean_ekf(:,T+1);
cov_eks{T+1} = cov_ekf{T+1};

mean_cks(:,T+1) = mean_ckf(:,T+1);
cov_cks{T+1} = cov_ckf{T+1};

mean_uks(:,T+1) = mean_ukf(:,T+1);
cov_uks{T+1} = cov_ukf{T+1};

for t = T+1:-1:2
  % Gibbs smoother
  mean_gibbs_smoother(:,t-1) = mean_gibbs_filter(:,t-1) ...
    + J_gibbs{t-1}*(mean_gibbs_smoother(:,t) - mxPred_gibbs(:,t));
  
  cov_gibbs_smoother{t-1} = cov_gibbs_filter{t-1} ...
    + J_gibbs{t-1}*(cov_gibbs_smoother{t} - SxPred_gibbs{t})*J_gibbs{t-1}';
  
  % EKS
  mean_eks(:,t-1) = mean_ekf(:,t-1) ...
    + J_ekf{t-1}*(mean_eks(:,t) - mxPred_ekf(:,t));
  
  cov_eks{t-1} = cov_ekf{t-1} ...
    + J_ekf{t-1}*(cov_eks{t} - SxPred_ekf{t})*J_ekf{t-1}';
  
  % CKS
  mean_cks(:,t-1) = mean_ckf(:,t-1) ...
    + J_ckf{t-1}*(mean_cks(:,t) - mxPred_ckf(:,t));
  
  cov_cks{t-1} = cov_ckf{t-1} ...
    + J_ckf{t-1}*(cov_cks{t} - SxPred_ckf{t})*J_ckf{t-1}';
  
  % URTSS
  mean_uks(:,t-1) = mean_ukf(:,t-1) ...
    + J_ukf{t-1}*(mean_uks(:,t) - mxPred_ukf(:,t));
  
  cov_uks{t-1} = cov_ukf{t-1} ...
    + J_ukf{t-1}*(cov_uks{t} - SxPred_ukf{t})*J_ukf{t-1}';
end

%% plotting

if plotOn
  
  % Gibbs stuff
  hor = 1:T+1;
  Axis = [0 T+1 -45 30];
  
  ff1 = [mean_gibbs_filter' + 2*sqrt(cell2mat(cov_gibbs_filter')); flipdim(mean_gibbs_filter' - 2*sqrt(cell2mat(cov_gibbs_filter')),1)];
  figure('units','pixel','outerposition',  [0 0 1200 800]);
  clf; hold on
  fill([hor';  flipdim(hor',1)], ff1, [7 7 8]/8, 'EdgeColor', [7 7 8]/8);
  plot(hor,mean_gibbs_smoother+2*sqrt(cell2mat(cov_gibbs_smoother)), 'color', [0 4 0]/8);
  %errorbar(hor, mean_gibbs_smoother, 2*sqrt(cell2mat(cov_gibbs_smoother)), 'color', [0 4 0]/8);
  plot(hor, x, 'r--','linewidth',5);
  plot(hor,mean_gibbs_smoother-2*sqrt(cell2mat(cov_gibbs_smoother)), 'color', [0 4 0]/8);
  grid on;
  set(gcf,'PaperSize', [10 5]);
  set(gcf,'PaperPosition',[0.1 0.1 10 5]);
  xlabel('time steps');
  ylabel('hidden states');
  axis(Axis)
  legend('Gibbs-filter','Gibbs-RTSS', 'ground truth','location','southwest');
  
  if printFig
    filename = 'gibbs_smoother';
    print_pdf(filename);
  end
  
  
  
  % CKF
  ff1 = [mean_ckf' + 2*sqrt(cell2mat(cov_ckf')); flipdim(mean_ckf' - 2*sqrt(cell2mat(cov_ckf')),1)];
  figure('units','pixel','outerposition',  [0 0 1200 800]);
  clf; hold on
  fill([hor';  flipdim(hor',1)], ff1, [7 7 8]/8, 'EdgeColor', [7 7 8]/8);
  plot(hor,mean_cks+ 2*sqrt(cell2mat(cov_cks)), 'color', [0 4 0]/8);
  plot(hor, x, 'r--','linewidth',5);
  plot(hor,mean_cks- 2*sqrt(cell2mat(cov_cks)), 'color', [0 4 0]/8);
  
  %   errorbar(hor, mean_cks, 2*sqrt(cell2mat(cov_cks)), 'color', [0 4 0]/8);
  
  grid on;
  set(gcf,'PaperSize', [10 5]);
  set(gcf,'PaperPosition',[0.1 0.1 10 5]);
  xlabel('time steps');
  ylabel('hidden states');
  axis(Axis)
  legend('CKF','CKS', 'ground truth','location','southwest');
  if printFig
    filename = 'cks';
    print_pdf(filename);
  end
  
  
  % EKF
  ff2 = [mean_ekf' + 2*sqrt(cell2mat(cov_ekf')); flipdim(mean_ekf' - 2*sqrt(cell2mat(cov_ekf')),1)];
  figure('units','pixel','outerposition',  [0 0 1200 800]);
  clf; hold on
  fill([hor';  flipdim(hor',1)], ff2, [7 7 8]/8, 'EdgeColor', [7 7 8]/8);
  %   errorbar(hor, mean_eks, 2*sqrt(cell2mat(cov_eks)), 'color', [0 4 0]/8);
  plot(hor,mean_eks+ 2*sqrt(cell2mat(cov_eks)), 'color', [0 4 0]/8);
  plot(hor, x, 'r--','linewidth',5);
  plot(hor,mean_eks- 2*sqrt(cell2mat(cov_eks)), 'color', [0 4 0]/8);
  
  grid on;
  set(gcf,'PaperSize', [10 5]);
  set(gcf,'PaperPosition',[0.1 0.1 10 5]);
  xlabel('time steps');
  ylabel('hidden states');
  axis(Axis)
  legend('EKF','EKS', 'ground truth','location','southwest');
  if printFig
    filename = 'eks';
    print_pdf(filename);
  end
  
  
  % UKF
  ff2 = [mean_ukf' + 2*sqrt(cell2mat(cov_ukf')); flipdim(mean_ukf' - 2*sqrt(cell2mat(cov_ukf')),1)];
  figure('units','pixel','outerposition',  [0 0 1200 800]);
  clf; hold on
  fill([hor';  flipdim(hor',1)], ff2, [7 7 8]/8, 'EdgeColor', [7 7 8]/8);
  %   errorbar(hor, mean_uks, 2*sqrt(cell2mat(cov_uks)), 'color', [0 4 0]/8);
  plot(hor,mean_uks+ 2*sqrt(cell2mat(cov_uks)), 'color', [0 4 0]/8);
  plot(hor, x, 'r--','linewidth',5);
  plot(hor,mean_uks- 2*sqrt(cell2mat(cov_uks)), 'color', [0 4 0]/8);
  grid on;
  set(gcf,'PaperSize', [10 5]);
  set(gcf,'PaperPosition',[0.1 0.1 10 5]);
  xlabel('time steps');
  ylabel('hidden states');
  axis(Axis)
  legend('UKF','URTSS', 'ground truth','location','southwest');
  if printFig
    filename = 'urtss';
    print_pdf(filename);
  end
  
  
end

%% some evaluation

fprintf('\n\n%s\n\n', '######### some analysis #########');

% RMSE
rmse_gibbs_filter = rmsefun(x, mean_gibbs_filter);
rmse_gibbs_smoother = rmsefun(x, mean_gibbs_smoother);
rmse_ekf = rmsefun(x, mean_ekf);
rmse_eks = rmsefun(x, mean_eks);
rmse_ckf = rmsefun(x, mean_ckf);
rmse_cks = rmsefun(x, mean_cks);
rmse_ukf = rmsefun(x, mean_ukf);
rmse_uks = rmsefun(x, mean_uks);

fprintf('\n');
fprintf('%s\n', 'RMSE values; smaller values are better:');
fprintf('%s %4.2e\n', 'Gibbs-filter: ', rmse_gibbs_filter);
fprintf('%s %4.2e\n', 'Gibbs-RTSS: ', rmse_gibbs_smoother);
fprintf('%s %4.2e\n', 'EKF: ', rmse_ekf);
fprintf('%s %4.2e\n', 'EKS: ', rmse_eks);
fprintf('%s %4.2e\n', 'CKF: ', rmse_ckf);
fprintf('%s %4.2e\n', 'CKS: ', rmse_cks);
fprintf('%s %4.2e\n', 'UKF: ', rmse_ukf);
fprintf('%s %4.2e\n', 'URTSS: ', rmse_uks);



% NLL (negative log likelihood)
nll_gibbs_filter = mean(nllfun(x, mean_gibbs_filter, cell2mat(cov_gibbs_filter)));
nll_gibbs_smoother = mean(nllfun(x, mean_gibbs_smoother, cell2mat(cov_gibbs_smoother)));
nll_ekf = mean(nllfun(x, mean_ekf, cell2mat(cov_ekf)));
nll_eks = mean(nllfun(x, mean_eks, cell2mat(cov_eks)));
nll_ckf = mean(nllfun(x, mean_ckf, cell2mat(cov_ckf)));
nll_cks = mean(nllfun(x, mean_cks, cell2mat(cov_cks)));
nll_ukf = mean(nllfun(x, mean_ukf, cell2mat(cov_ukf)));
nll_uks = mean(nllfun(x, mean_uks, cell2mat(cov_uks)));

fprintf('\n');
fprintf('%s\n', 'negative log-likelihoods (per step in time series); smaller values are better:');
fprintf('%s %4.2e\n', 'Gibbs-filter: ', nll_gibbs_filter);
fprintf('%s %4.2e\n', 'Gibbs-RTSS: ', nll_gibbs_smoother);
fprintf('%s %4.2e\n', 'EKF: ', nll_ekf);
fprintf('%s %4.2e\n', 'EKS: ', nll_eks);
fprintf('%s %4.2e\n', 'CKF: ', nll_ckf);
fprintf('%s %4.2e\n', 'CKS: ', nll_cks);
fprintf('%s %4.2e\n', 'UKF: ', nll_ukf);
fprintf('%s %4.2e\n', 'URTSS: ', nll_uks);
