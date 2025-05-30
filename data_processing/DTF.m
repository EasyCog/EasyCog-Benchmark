function [gamma2] = DTF(ts,low_freq,high_freq,p,fs)
% DTF - perform Directed Transfer Function analysis among multi-channel time series. 
%
% Usage: gamma2 = DTF(ts,low_freq,high_freq,p,fs)
%
% Input: ts - the time series where each column is the temporal data from a
%           single channel
%           low_freq - the lowest frequency bound to compute the DTF
%           high_freq - the highest frequency bound to compute the DTF
%           p - the order of the MVAR model
%           fs - the sampling frequency 
%
% Output: gamma2 - the computed DTF values
%
% Description: This program calculates the DTF values for a given frequency
%              range for the input time series. The output is in the form
%              gamma2(a,b,c) where a = the sink channel, b = the source
%              channel, c = the frequency index.
%
% ARfit Package:
% The ARfit package is used in DTF computation. 
% See below for detailed description of the ARfit package:
% A. Neumaier and T. Schneider, 2001: Estimation of parameters and eigenmodes of 
% multivariate autoregressive models. ACM Trans. Math. Softw., 27, 27?57.
% T. Schneider and A. Neumaier, 2001: Algorithm 808: ARfit-A Matlab package for the 
% estimation of parameters and eigenmodes of multivariate autoregressive models. 
% ACM Trans. Math. Softw., 27, 58?65. 
% http://www.gps.caltech.edu/~tapio/arfit/' 
%
% Program Authors: Lei Ding and Christopher Wilke, University of Minnesota, USA
%
% User feedback welcome: e-mail: econnect@umn.edu
%

% License
% ==============================================================
% This program is part of the eConnectome.
% 
% Copyright (C) 2010 Regents of the University of Minnesota. All rights reserved.
% Correspondence: binhe@umn.edu
% Web: econnectome.umn.edu
%
% This program is free software for academic research: you can redistribute it and/or modify
% it for non-commercial uses, under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program. If not, see http://www.gnu.org/copyleft/gpl.html.
%
% This program is for research purposes only. This program
% CAN NOT be used for commercial purposes. This program 
% SHOULD NOT be used for medical purposes. The authors 
% WILL NOT be responsible for using the program in medical
% conditions.
% ==========================================

% Revision Logs
% ==========================================
%
% Yakang Dai, 01-Mar-2010 15:20:30
% Release Version 1.0 beta 
%
% ==========================================

% Default sampling rate is 400 Hz
if nargin < 5
    fs = 400;
end

% The number of frequencies to compute the DTF over
tot_range = [low_freq:high_freq];
nfre = length(tot_range);

% The number of channels in the time series
nchan = size(ts,2);

% The sampling period
dt = 1/fs;

% Create the MVAR matrix for the time series
[w,A] = arfit(ts,p,p);

% Rearrange the format of the MVAR matrix
B = [];
B(:,:,1) = -eye(nchan);
for i=1:nchan
    for j=1:nchan
%         dc(i,j) = sum(A(i,j:nchan:nchan*p).*A(i,j:nchan:nchan*p));
        B(i,j,2:p+1) = A(i,j:nchan:nchan*p);
    end
end

% Calculate the non-normalized DTF value
theta2 = [];
for k = 1:nfre
    Af = zeros(nchan,nchan);
    fre = tot_range(k);
    for i = 1:nchan
        for j = 1:nchan
            for h = 1:p+1
                Af(i,j) = Af(i,j)-B(i,j,h)*exp(-pi*fre*dt*(h-1)*2i);
            end
        end
    end
    dett2 = det(Af);
    dett2 = dett2.*conj(dett2);
    for i = 1:nchan
        for j = 1:nchan
            Apf = Af;
            Apf(:,i) = [];
            Apf(j,:) = [];
            det2 = det(Apf);
            det2 = det2.*conj(det2);
            theta2(i,j,k) = det2/dett2;
        end
    end
end

% Calculate the normalized DTF values
gamma2 = [];
for k=1:nfre
    for i=1:nchan
        for j=1:nchan
            gamma2(i,j,k) = theta2(i,j,k) / sum(theta2(i,:,k),2);
        end
    end
end

function [R, scale]=arqr(v, p, mcor)
%ARQR	QR factorization for least squares estimation of AR model.
%
%  [R, SCALE]=ARQR(v,p,mcor) computes the QR factorization needed in
%  the least squares estimation of parameters of an AR(p) model. If
%  the input flag mcor equals one, a vector of intercept terms is
%  being fitted. If mcor equals zero, the process v is assumed to have
%  mean zero. The output argument R is the upper triangular matrix
%  appearing in the QR factorization of the AR model, and SCALE is a
%  vector of scaling factors used to regularize the QR factorization.
%
%  ARQR is called by ARFIT. 
%
%  See also ARFIT.

%  Modified 29-Dec-99
%           24-Oct-10 Tim Mullen (added support for multiple realizatons)
%
%  Author: Tapio Schneider
%          tapio@gps.caltech.edu

  % n:   number of time steps (per realization)
  % m:   number of variables (dimension of state vectors) 
  % ntr: number of realizations (trials)
  [n,m,ntr] = size(v);     

  ne    = ntr*(n-p);            % number of block equations of size m
  np    = m*p+mcor;             % number of parameter vectors of size m

  % If the intercept vector w is to be fitted, least squares (LS)
  % estimation proceeds by solving the normal equations for the linear
  % regression model
  %
  %                  v(k,:)' = Aaug*u(k,:)' + noise(C)        (1)
  %
  % with Aaug=[w A] and `predictors' 
  %
  %              u(k,:) = [1 v(k-1,:) ...  v(k-p,:)].         (2a) 
  %
  % If the process mean is taken to be zero, the augmented coefficient
  % matrix is Aaug=A, and the regression model
  %
  %                u(k,:) = [v(k-1,:) ...  v(k-p,:)]          (2b)
  %
  % is fitted. 
  % The number np is the dimension of the `predictors' u(k). 
  %
  % If multiple realizations are given (ntr > 1), they are appended
  % as additional ntr-1 blocks of rows in the normal equations (1), and
  % the 'predictors' (2) correspondingly acquire additional row blocks.
  
  % Initialize the data matrix K (of which a QR factorization will be computed)
  K = zeros(ne,np+m);                 % initialize K
  if (mcor == 1)
    % first column of K consists of ones for estimation of intercept vector w
    K(:,1) = ones(ne,1);
  end
  
  % Assemble `predictors' u in K 
  for itr=1:ntr
    for j=1:p
      K((n-p)*(itr-1) + 1 : (n-p)*itr, mcor+m*(j-1)+1 : mcor+m*j) = ...
          squeeze(v(p-j+1:n-j, :, itr));
    end
    % Add `observations' v (left hand side of regression model) to K
    K((n-p)*(itr-1) + 1 : (n-p)*itr, np+1 : np+m) = squeeze(v(p+1:n, :, itr));
  end
  
  % Compute regularized QR factorization of K: The regularization
  % parameter delta is chosen according to Higham's (1996) Theorem
  % 10.7 on the stability of a Cholesky factorization. Replace the
  % regularization parameter delta below by a parameter that depends
  % on the observational error if the observational error dominates
  % the rounding error (cf. Neumaier, A. and T. Schneider, 2001:
  % "Estimation of parameters and eigenmodes of multivariate
  % autoregressive models", ACM Trans. Math. Softw., 27, 27--57.).
  q     = np + m;             % number of columns of K
  delta = (q^2 + q + 1)*eps;  % Higham's choice for a Cholesky factorization
  scale = sqrt(delta)*sqrt(sum(K.^2));   
  R     = triu(qr([K; diag(scale)]));

function [sbc, fpe, logdp, np] = arord(R, m, mcor, ne, pmin, pmax)
%ARORD	Evaluates criteria for selecting the order of an AR model.
%
%  [SBC,FPE]=ARORD(R,m,mcor,ne,pmin,pmax) returns approximate values
%  of the order selection criteria SBC and FPE for models of order
%  pmin:pmax. The input matrix R is the upper triangular factor in the
%  QR factorization of the AR model; m is the dimension of the state
%  vectors; the flag mcor indicates whether or not an intercept vector
%  is being fitted; and ne is the number of block equations of size m
%  used in the estimation. The returned values of the order selection
%  criteria are approximate in that in evaluating a selection
%  criterion for an AR model of order p < pmax, pmax-p initial values
%  of the given time series are ignored.
%
%  ARORD is called by ARFIT. 
%	
%  See also ARFIT, ARQR.

%  For testing purposes, ARORD also returns the vectors logdp and np,
%  containing the logarithms of the determinants of the (scaled)
%  covariance matrix estimates and the number of parameter vectors at
%  each order pmin:pmax.

%  Modified 17-Dec-99
%  Author: Tapio Schneider
%          tapio@gps.caltech.edu
  
  imax 	  = pmax-pmin+1;        % maximum index of output vectors
  
  % initialize output vectors
  sbc     = zeros(1, imax);     % Schwarz's Bayesian Criterion
  fpe     = zeros(1, imax);     % log of Akaike's Final Prediction Error
  logdp   = zeros(1, imax);     % determinant of (scaled) covariance matrix
  np      = zeros(1, imax);     % number of parameter vectors of length m
  np(imax)= m*pmax+mcor;

  % Get lower right triangle R22 of R: 
  %
  %   | R11  R12 |
  % R=|          |
  %   | 0    R22 |
  %
  R22     = R(np(imax)+1 : np(imax)+m, np(imax)+1 : np(imax)+m);

  % From R22, get inverse of residual cross-product matrix for model
  % of order pmax
  invR22  = inv(R22);
  Mp      = invR22*invR22';
  
  % For order selection, get determinant of residual cross-product matrix
  %       logdp = log det(residual cross-product matrix)
  logdp(imax) = 2.*log(abs(prod(diag(R22))));

  % Compute approximate order selection criteria for models of 
  % order pmin:pmax
  i = imax;
  for p = pmax:-1:pmin
    np(i)      = m*p + mcor;	% number of parameter vectors of length m
   if p < pmax
      % Downdate determinant of residual cross-product matrix
      % Rp: Part of R to be added to Cholesky factor of covariance matrix
      Rp       = R(np(i)+1:np(i)+m, np(imax)+1:np(imax)+m);

      % Get Mp, the downdated inverse of the residual cross-product
      % matrix, using the Woodbury formula
      L        = chol(eye(m) + Rp*Mp*Rp')';
      N        = L \ Rp*Mp;
      Mp       = Mp - N'*N;

      % Get downdated logarithm of determinant
      logdp(i) = logdp(i+1) + 2.* log(abs(prod(diag(L))));
   end

   % Schwarz's Bayesian Criterion
   sbc(i) = logdp(i)/m - log(ne) * (ne-np(i))/ne;

   % logarithm of Akaike's Final Prediction Error
   fpe(i) = logdp(i)/m - log(ne*(ne-np(i))/(ne+np(i)));

   % Modified Schwarz criterion (MSC):
   % msc(i) = logdp(i)/m - (log(ne) - 2.5) * (1 - 2.5*np(i)/(ne-np(i)));

   i      = i-1;                % go to next lower order
end

function [w, A, C, sbc, fpe, th]=arfit(v, pmin, pmax, selector, no_const)
%ARFIT	Stepwise least squares estimation of multivariate AR model.
%
%  [w,A,C,SBC,FPE,th]=ARFIT(v,pmin,pmax) produces estimates of the
%  parameters of an m-variate AR model of order p,
%
%      v(k,:)' = w' + A1*v(k-1,:)' +...+ Ap*v(k-p,:)' + noise(C),
%
%  where w is the (m x 1) intercept vector, A1, ..., Ap are (m x m)
%  coefficient matrices, and C is a (m x m) noise covariance
%  matrix. The estimated order p lies between pmin and pmax and is
%  chosen as the optimizer of Schwarz's Bayesian Criterion. 
% 
%  The input matrix v must contain the time series data, with
%  columns v(:,l) representing m variables l=1,...,m and rows
%  v(k,:) representing n observations at different (equally
%  spaced) times k=1,..,n. Optionally, v can have a third
%  dimension, in which case the matrices v(:,:, itr) represent  
%  the realizations (e.g., measurement trials) itr=1,...,ntr of the
%  time series. ARFIT returns least squares estimates of the
%  intercept vector w, of the coefficient matrices A1,...,Ap (as
%  A=[A1 ... Ap]), and of the noise covariance matrix C.
%
%  As order selection criteria, ARFIT computes approximations to
%  Schwarz's Bayesian Criterion and to the logarithm of Akaike's Final
%  Prediction Error. The order selection criteria for models of order
%  pmin:pmax are returned as the vectors SBC and FPE.
%
%  The matrix th contains information needed for the computation of
%  confidence intervals. ARMODE and ARCONF require th as input
%  arguments.
%       
%  If the optional argument SELECTOR is included in the function call,
%  as in ARFIT(v,pmin,pmax,SELECTOR), SELECTOR is used as the order
%  selection criterion in determining the optimum model order. The
%  three letter string SELECTOR must have one of the two values 'sbc'
%  or 'fpe'. (By default, Schwarz's criterion SBC is used.) If the
%  bounds pmin and pmax coincide, the order of the estimated model
%  is p=pmin=pmax. 
%
%  If the function call contains the optional argument 'zero' as the
%  fourth or fifth argument, a model of the form
%
%         v(k,:)' = A1*v(k-1,:)' +...+ Ap*v(k-p,:)' + noise(C) 
%
%  is fitted to the time series data. That is, the intercept vector w
%  is taken to be zero, which amounts to assuming that the AR(p)
%  process has zero mean.
%
%  Modified 14-Oct-00
%           24-Oct-10 Tim Mullen (added support for multiple realizatons)
%
%  Authors: Tapio Schneider
%           tapio@gps.caltech.edu
%
%           Arnold Neumaier
%           neum@cma.univie.ac.at

  % n:   number of time steps (per realization)
  % m:   number of variables (dimension of state vectors) 
  % ntr: number of realizations (trials)
  [n,m,ntr]   = size(v);     

  if (pmin ~= round(pmin) | pmax ~= round(pmax))
    error('Order must be integer.');
  end
  if (pmax < pmin)
    error('PMAX must be greater than or equal to PMIN.')
  end

  % set defaults and check for optional arguments
  if (nargin == 3)              % no optional arguments => set default values
    mcor       = 1;               % fit intercept vector
    selector   = 'sbc';	          % use SBC as order selection criterion
  elseif (nargin == 4)          % one optional argument
    if strcmp(selector, 'zero')
      mcor     = 0;               % no intercept vector to be fitted
      selector = 'sbc';	          % default order selection 
    else
      mcor     = 1; 		  % fit intercept vector
    end
  elseif (nargin == 5)          % two optional arguments
    if strcmp(no_const, 'zero')
      mcor     = 0;               % no intercept vector to be fitted
    else
      error(['Bad argument. Usage: ', ...
	     '[w,A,C,SBC,FPE,th]=AR(v,pmin,pmax,SELECTOR,''zero'')'])
    end
  end

  ne  	= ntr*(n-pmax);         % number of block equations of size m
  npmax	= m*pmax+mcor;          % maximum number of parameter vectors of length m

  if (ne <= npmax)
    error('Time series too short.')
  end

  % compute QR factorization for model of order pmax
  [R, scale]   = arqr(v, pmax, mcor);

  % compute approximate order selection criteria for models 
  % of order pmin:pmax
  [sbc, fpe]   = arord(R, m, mcor, ne, pmin, pmax);

  % get index iopt of order that minimizes the order selection 
  % criterion specified by the variable selector
  [val, iopt]  = min(eval(selector)); 

  % select order of model
  popt         = pmin + iopt-1; % estimated optimum order 
  np           = m*popt + mcor; % number of parameter vectors of length m

  % decompose R for the optimal model order popt according to 
  %
  %   | R11  R12 |
  % R=|          |
  %   | 0    R22 |
  %
  R11   = R(1:np, 1:np);
  R12   = R(1:np, npmax+1:npmax+m);    
  R22   = R(np+1:npmax+m, npmax+1:npmax+m);

  % get augmented parameter matrix Aaug=[w A] if mcor=1 and Aaug=A if mcor=0
  if (np > 0)   
    if (mcor == 1)
      % improve condition of R11 by re-scaling first column
      con 	= max(scale(2:npmax+m)) / scale(1); 
      R11(:,1)	= R11(:,1)*con; 
    end;
    Aaug = (R11\R12)';
    
    %  return coefficient matrix A and intercept vector w separately
    if (mcor == 1)
      % intercept vector w is first column of Aaug, rest of Aaug is 
      % coefficient matrix A
      w = Aaug(:,1)*con;        % undo condition-improving scaling
      A = Aaug(:,2:np);
    else
      % return an intercept vector of zeros 
      w = zeros(m,1);
      A = Aaug;
    end
  else
    % no parameters have been estimated 
    % => return only covariance matrix estimate and order selection 
    % criteria for ``zeroth order model''  
    w   = zeros(m,1);
    A   = [];
  end
  
  % return covariance matrix
  dof   = ne-np;                % number of block degrees of freedom
  C     = R22'*R22./dof;        % bias-corrected estimate of covariance matrix
  
  % for later computation of confidence intervals return in th: 
  % (i)  the inverse of U=R11'*R11, which appears in the asymptotic 
  %      covariance matrix of the least squares estimator
  % (ii) the number of degrees of freedom of the residual covariance matrix 
  invR11 = inv(R11);
  if (mcor == 1)
    % undo condition improving scaling
    invR11(1, :) = invR11(1, :) * con;
  end
  Uinv   = invR11*invR11';
  th     = [dof zeros(1,size(Uinv,2)-1); Uinv];



