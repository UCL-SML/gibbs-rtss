function [x, P, x1, Px, mz, Pz, J] = ekf(fstate, x, P, hmeas, z, Q, R, u)
% EKF   Extended Kalman Filter for nonlinear dynamic systems
% [x, P] = ekf(f,x,P,h,z,Q,R) returns state estimate, x and state covariance, P
% for nonlinear dynamic system:
%           x_k+1 = f(x_k) + w_k
%           z_k   = h(x_k) + v_k
% where w ~ N(0,Q) meaning w is gaussian noise with covariance Q
%       v ~ N(0,R) meaning v is gaussian noise with covariance R
% Inputs:   f: function handle for f(x)
%           x: "a priori" state estimate (mean)
%           P: "a priori" estimated state covariance
%           h: function handle for h(x)
%           z: current measurement
%           Q: process noise covariance
%           R: measurement noise covariance
% Output:   x: "a posteriori" state estimate (mean)
%           P: "a posteriori" state covariance
%
%
% By Yi Cao at Cranfield University, 02/01/2008
%
% Extended to smoothing and fixed some problems
% Marc Deisenroth, 2010-08-02
%

Pin = P;
if exist('u','var') % checks for control signals
  [x1,A] = jacfd(fstate,x,u); %nonlinear update and linearization at current state
else
  [x1,A] = jacfd(fstate,x);   %nonlinear update and linearization at current state
end
P = A*P*A'+ Q;              %partial update
Px = P;

if nargout > 6
  % for smoothing
  crossterm = Pin*A';         % cov(x_t,x_{t+1})
  J = crossterm / Px;
end

[z1,H] = jacfd(hmeas,x1);    % nonlinear measurement and linearization
P12 = P*H';                  % cross covariance cov(x,z)
% K = P12*inv(H*P12+R);      % Kalman filter gain
% x = x1+K*(z-z1);           % state estimate
% P = P-K*P12';              % state covariance matrix
mz = z1;                     % mean of predicted measurement
Pz = H*P12 + R;              % covariance of predicted measurement
R1 = chol(Pz);               % Cholesky factorization
U = P12/R1;                  % K=U/R'; Faster because of back substitution
x = x1 + U*(R1'\(z-z1));     % Back substitution to get state update
P = P - U*U';                % Covariance update, U*U'=P12/R/R'*P12'=K*P12



function [z, A] = jacfd(fun,x,u)
% JACfd Jacobian through finite differences
% [z J] = jacfd(f,x)
% z = f(x)
% J = f'(x)
%
if exist('u','var')
  z = fun(x,u);
else
  z = fun(x);
end
n = numel(x);
m = numel(z);
A = zeros(m,n);
h = 1e-4; % difference step

for k=1:n
  x1=x;
  x1(k)=x1(k) + h;
  if exist('u','var')
    A(:,k)=(fun(x1,u)-z)/h;
  else
    A(:,k)=(fun(x1)-z)/h;
  end
end