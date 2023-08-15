function S = variog(x,y,trendfun,nrbins,maxdist)

% Description:
%   variog calculates the GNSS data experimental variogram 
%   based variogram m-file
%
% Input:
%   x - array with GNSS [longitude latitude] coordinates, 
%       a m*2 matrix. m is number of dimensions GNSS site.
%   y - GNSS series, a T*m matrix.T is number of dimensions time,
%   trendfun - choosed trend function to remove data trend
%              ,mostly used function is constant, linear 
%              and triple etc.
%   nrbins - number bins the distance should be grouped into
%            (default = 20)
%   maxdist - maximum distance for variogram calculation
%             (default = maximum distance in the dataset / 2)
%
% Output:
%   S - structure array with distance and gamma - vector

% error checking
if size(y,2) ~= size(x,1)
    error('x and y must have the same number of rows!');
end

% change coordinate into normal distribution
x1 = x(:,1); y1 = x(:,2);
mxlength = length(x1);
mx1 = mean(x1); sx1 = std(x1);
my1 = mean(y1); sy1 = std(y1);
x1 = (x1 - repmat(mx1,mxlength,1))./repmat(sx1,mxlength,1);
y1 = (y1 - repmat(my1,mxlength,1))./repmat(sy1,mxlength,1);
x = [x1,y1];

if isempty(nrbins)
    nrbins = 20;
end

if isempty(maxdist)
    minx = min(x,[],1);
    maxx = max(x,[],1);
    maxdist = sqrt(sum((maxx-minx).^2))/2;
end

% remove trend
T = size(y,1);
num_point = size(x,1);
res = nan(size(y));
for t=1:T
    idex = any(isnan(y(t,:)'),2); %check NaN value
    ytmp = y(t,:)';%restore time t value
    xtmp = x;
    ytmp(idex,:) = [];%delete NaN value
    xtmp(idex,:) = [];
    F = feval(trendfun,xtmp(:,1),xtmp(:,2));%Trend Design Matrix
    beta = (F'*F)\F'*ytmp;
    delta = ytmp - F*beta;
    res(t,~idex) = delta;
end

% calculate the exp-variogram value
num = num_point*(num_point+1)/2;
d = nan(num,1);%distance
val = nan(T,num);
ll = 0;
for k = 1:num_point
    ll = ll(end)+(1:num_point+1-k);
    dis = repmat(x(k,:),num_point+1-k,1) - x(k:num_point,:);
    d(ll,:) = sqrt(dis(:,1).^2 + dis(:,2).^2);
    val(:,ll) = (repmat(res(:,k),1,num_point+1-k)...
        - res(:,k:num_point)).^2;
end

d(d>=maxdist,:) = [];
val(:,d>=maxdist)=[];
edges = linspace(0,maxdist,nrbins+1);
Sval = zeros(nrbins,1);
Sd = zeros(nrbins,1);
numcount = zeros(nrbins,1);
for i=1:nrbins
    semi_varig_tmp = val(:,d>edges(i)& d<=edges(i+1));
    idex = isnan(semi_varig_tmp);
    numcount(i,1) = sum(sum(~idex));
    semi_varig_tmp = semi_varig_tmp(:);
    semi_varig_tmp(idex) = [];
    std_semi_varig_tmp = std(semi_varig_tmp);
    idex = abs(semi_varig_tmp)>3*std_semi_varig_tmp;
    semi_varig_tmp(idex) = [];
%     semi_varig_tmp = sort(semi_varig_tmp);
%     istart = ceil(0.3*length(semi_varig_tmp));
%     iend = ceil(0.8*length(semi_varig_tmp));
%     semi_varig_tmp(:,[1:istart iend:end]) = [];
    Sval(i,1) = mean(semi_varig_tmp)/2;
    Sd(i,1) = (edges(i)+edges(i+1))/2;
end

S.distance = Sd;
S.val = Sval;
S.num = numcount;



