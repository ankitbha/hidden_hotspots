function H = SpatialFiled(x,y,S,proportion)

% Description:
%   Compute Spatial Filed of Kriged Kalman Filter
%
% Input:
%   x - array with longitude coordinates.
%   y - array with latitude coordinates.
%   S - Parameter of Semi-Variogram Value Fit
%
% Outout:
%   H - Spatial Filed

% error checking
if size(y,1) ~= size(x,1)
    error('x and y must have the same number of rows!');
end

R = ComputeCov(S,x,y);

% trend field
F = feval(S.trendfun,x,y);
q = size(F,2);

n = size(R,1);
InvR = eye(n)/R;
A = (F'*InvR*F)\F'*InvR;%Trend Desigen Matrix
B = InvR - InvR*F*A;%bending energy matrix
[U,E] = svd(B);%spectral decomposition of B
U=fliplr(U);E=fliplr(rot90(E));
e = diag(E);%eigen value

% dimensionality reduction
[e,ij] = sort(e,'ascend');
ij = n*repmat(ij'-1,n,1)+repmat((1:n)',1,n);
U = U(ij);
for i = 1:n
    if((sum(e(1:i))/sum(e)) >= proportion)
        p = i;
        break;
    end
end

% construct spatial filed
e = e(1:p,:);
E = repmat(e',n,1);
U = U(:,1:p);
H2 = (R*U);%principle fields
H2(:,1:q) = [];
H = [F,H2];

function R = ComputeCov(S,x,y,xi,yi)

% Description:
%   Compute covariance matrix according to Semi-Variogram function
%
% Input:
%   S - Parameter ofSemi-Variogram function
%   x,y - konwn point coordinates
%   xi,yi - unkonwn point coordinates
%
% Output:
%   R - covariance matrix

% check Input value
if nargin ==3
    xi = x;  yi = y;
elseif nargin == 5
else
    error('Number of Input Parameter are not correct!');
end

if length(x) ~= length(y)
    error('x and y must have the same number of rows');
end

N1 = length(x);%number of konwn point
N2 = length(xi);%number of unkonwn point

%compute covariance matrix
mzmax = N1*N2;
ij = nan(mzmax,2);
d = nan(mzmax,1);
ll = 0;
for k=1:N1
    ll = ll(end)+(1:N2);
    ij(ll,:) = [repmat(k,N2,1) (1:N2)'];
    d(ll,:) = sqrt((repmat(x(k,1),N2,1) - xi).^2 +...
        (repmat(y(k,1),N2,1) - yi).^2);
end

r = zeros(mzmax,1);

% range of Semi-Variogram function
if strcmp(S.model, 'spherical')
    a = S.range;
elseif strcmp(S.model, 'exponential')
    a = S.range;
elseif strcmp(S.model, 'gaussian')
    a = S.range;
end

% a = S.range;

idx = d <= a;
b = [S.range,S.sill];
r(idx) = feval(S.func,b,a) - feval(S.func,b,d(idx));
R = sparse(ij(idx,1),ij(idx,2),r(idx),N1,N2);

