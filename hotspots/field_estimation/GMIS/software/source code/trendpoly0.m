function F = trendpoly0(x,y)
%趋势为常数
%输入值：
%       x为横坐标
%       y为纵坐标
%输出值：
%       F为趋势矩阵
x = x/10^6;
y = y/10^6;
n = length(x);
if length(x) ~= length(y)
    errordlg('横纵坐标维数应该相等');
    return;
end

F = ones(n,1);
end