function obs = EMEst_filter(Zt,Ht,F,alpha0,P0,R,Q,iters,direction)

% Description:
%   Expectation Maximization(EM) Algorithm with Kriged Kalman
%   Filter to filter observed data and interpolate missing data
%
% Input:
%   Zt - observed values, a n*m matrix, n is the length of 
%        observed time, m is the length of observed site.
%   Ht - Spatial Filed of Kriged Kalman Filter or a unit matrix
%   F - state transition matrix
%   alpha0 - initial state vector of Kriged Kalman Filter
%   P0 - covariance matrix of initial state vector
%   R - observation noise covariance matrix
%   Q - system state noise covariance matrix
%   iters - EM Algorithm iteration number
%
% Output:
%   obs - filtered and interpolated missing data

[n,m] = size(Zt);%size of observation matrix
p = size(Ht,2);%dimension of spatial filed
if direction == 1
    str_direciton = 'EM in N direciton:  ';
elseif direction == 2
    str_direciton = 'EM in E direciton:  ';
elseif direction == 3
    str_direciton = 'EM in U direciton:  ';
else
    error('Error in the direction!');
end
obs = zeros(n,m);
for j=1:iters
    alpha_t = alpha0;
    Pt = P0;
    A = zeros(p,p);
    B = zeros(p,p);
    C = zeros(p,p);
    Rn = zeros(m,m); 
    
    str_process = sprintf('the %dth Kalman Filter,totally %d/%d',j,j,iters);
    str = [str_direciton,str_process];
    h = waitbar(0,str);
    for i=1:n
        if ~ishandle(h)
            obs = [];
            return;
        end
        waitbar(i/n,h);
        %change missing value and design matrix into 0
        II = any(isnan(Zt(i,:)'),2);
        Z = Zt(i,:)';  Z(II,:)=0;
        H = Ht;   H(II,:)=0;
        
        Rc = R;
        II_F = ~II;
        II = double(II);
        II_F = double(II_F);
        idex = II'*II + II_F'*II_F;
        idex = ~(logical(idex));
        Rc(idex) = 0;
        
        %Kalman Filter
        alpha_ = F*alpha_t;%one-step forecast state value
        P_ = F*Pt*F' + Q;%covariance matrix of one-step forecast value
        K = P_*H'/(H*P_*H' + Rc);%gain matrix
        alpha_t1 = alpha_ + K*(Z - H*alpha_);%filtered state value
        Pt1 = P_ - K*H*P_;%covariance matrix of filtered state value
        Ptt1 = (eye(p) - K*H)*F*Pt;%covariance matrix between forecast filtered state value
        
        obs(i,:) = (Ht*alpha_t1)';
        
        A = A + Pt + alpha_t*alpha_t';
        B = B + Ptt1 + alpha_t1*alpha_t';
        C = C + Pt1 + alpha_t1*alpha_t1';

        Rc = zeros(m,m); 
        II = double(II); II = II*II'; II = logical(II);
        Rc(II) = R(II);
        Rn = Rn +...
            ((Z-H*alpha_t)*(Z-H*alpha_t)'+...
            H*Pt*H' + Rc)/n;
        
        alpha_t = alpha_t1;
        Pt = Pt1;
    end
    close(h)
    
    %update new Kalman parameter
    F = B/A;
    Q = (C - F*B')/n;
    R = Rn;
end

%{
str_process = sprintf('Interpolate missing data... ing');
str = [str_direciton,str_process];
h = waitbar(0,str);

alpha_t = alpha0;
Pt = P0;
%robust Kalman FIlter
for i=1:n
    if ~ishandle(h)
        obs = [];
        return;
    end
    waitbar(i/n,h);
    %change missing value and design matrix into 0
    II = any(isnan(Zt(i,:)'),2);
    Z = Zt(i,:)';  Z(II,:)=0;
    H = Ht;   H(II,:)=0;
    
    %change R into zeros forms
    Rc = R;
    II_F = ~II;
    II = double(II);
    II_F = double(II_F);
    idex = II'*II + II_F'*II_F;
    idex = ~(logical(idex));
    Rc(idex) = 0;
    
    Rc_ = Rc;%initialize equivalent R matrix
    alpha_old = 10000*ones(p,1);%old state value
    
    %robust Kalman Filter
    alpha_ = F*alpha_t;
    P_ = F*Pt*F' + Q;
    while 1
        K = P_*H'/(H*P_*H' + Rc);
        alpha_t1 = alpha_ + K*(Z - H*alpha_);
        Pt1 = (eye(p) - K*H)*Pt*(eye(p) - K*H)' + K*Rc*K';

        dx = alpha_old - alpha_t1;
        alpha_old = alpha_t1;
        if(sqrt((dx'*dx)) < 1E-6)
            break;
        end
        
        V = H*alpha_t1 - Z;
        Qvv = Rc - H*Pt1*H';
        mv = sqrt(diag(Qvv));
        absV = V./mv;
        [~,idex] = max(absV);
        Rc_(idex,idex) = Rc(idex,idex)/...
            Wi(absV(idex),flag_robust);
        Rc = Rc_;
    end
    Pt1 = (eye(p) - K*H)*Pt*(eye(p) - K*H)' + K*Rc*K';
    
    alpha_t = alpha_t1;
    Pt = Pt1;
    
    obs(i,:) = (Ht*alpha_t1)';
end
close(h);

%IGGIII Equivalent Weight Functions
function w = Wi(absV,flag)
if flag == false
    w = 1;
else
    d = (1.5-absV) / (2.5-1.5);
    if(absV <= 1.5)
        w = 1;
    elseif(absV > 1.5 && absV < 2.5) 
        w = (1.5/absV)*d*d;
    else
        w = 1e-3;
    end
end
%}




