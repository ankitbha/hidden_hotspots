function F = trendpoly1(x,y)
%趋势为线性
%输入值：
%       x为横坐标
%       y为纵坐标
%输出值：
%       F为趋势矩阵

n = length(x);
if length(x) ~= length(y)
    errordlg('横纵坐标维数应该相等');
    return;
end

F = [x y ones(n,1)];
end