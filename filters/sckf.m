function [xe Se xp Sp zp Sz J] = sckf(xkk, S, systemFct, measFct, z, Q, R, u)

% cubature Kalman smoother
%
% last modified: 2010-08-12, Marc Deisenroth


Skk =  chol(S)';
Qsqrt = chol(Q)';
Rsqrt = chol(R)';


%% time update

%%%========================================================================
%%% Generate a set of Cubature Points
%%%========================================================================

nx = length(xkk); % state vector dimension

nPts = 2*nx;        % No. of Cubature Points

CPtArray = sqrt(nPts/2)*[eye(nx) -eye(nx)];



Xi_init =  repmat(xkk,1,nPts) + Skk*CPtArray; % cubature points

if exist('u','var')
  Xi = systemFct(Xi_init, repmat(u,1,nPts));
else
  Xi = systemFct(Xi_init);
end

xkk1 = sum(Xi,2)/nPts;      % predicted state

% predictive state distribution
xp = xkk1;
% Sp =  Xi*Xi'/nPts - xp*xp' + Q; % predicted covariance
X = (Xi-repmat(xkk1,1,nPts))/sqrt(nPts);
[foo,Skk1] = qr([X Qsqrt]',0);
Skk1 = Skk1';
Sp = Skk1*Skk1';

if nargout > 6
  % compute J:
  exxCubPt = bsxfun(@minus,Xi_init,xkk); % error of initial cubature points (in relation to prior mean)
  exCubPt = bsxfun(@minus,Xi, xp);       % error of predicted cubature points (in relation to pred. mean)
  crossterm = exxCubPt*exCubPt'/nPts;    % divide by the number of cubature points -> cov[x_t, x_{t+1}];
  J = crossterm / Sp;                    % required for smoothing
end


%% measurement update


%%%========================================================================
%%% Genrate a set of Cubature Points
%%%========================================================================

nz = length(z); % measurement vector dimension

nPts = 2*nx;

CPtArray = sqrt(nPts/2)*[eye(nx) -eye(nx)];

%%%========================================================================

Xi =  repmat(xkk1,1,nPts) + Skk1*CPtArray;

Zi = measFct(Xi);

zkk1 = sum(Zi,2)/nPts;   % predicted Measurement

X = (Xi-repmat(xkk1,1,nPts))/sqrt(nPts); 

Z = (Zi-repmat(zkk1,1,nPts))/sqrt(nPts); 

[foo,S] = qr([Z Rsqrt; X zeros(nx,nz)]',0);

S = S';

A = S(1:nz,1:nz);   % Square-root Innovations Covariance

B = S(nz+1:end,1:nz);

C = S(nz+1:end,nz+1:end);

G = B/A;          % Cubature Kalman Gain

xkk = xkk1 + G*(z-zkk1);  % filtered mean

Skk = C;

xe = xkk;
Se = Skk*Skk';
zp = zkk1;
Sz = A*A';