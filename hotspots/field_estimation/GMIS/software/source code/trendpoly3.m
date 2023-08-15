function F = trendpoly3(x,y)
%趋势为线性
%输入值：
%       x为横坐标
%       y为纵坐标
%输出值：
%       F为趋势矩阵
if length(x) ~= length(y)
    errordlg('横纵坐标维数应该相等');
    return;
end
m=length(x);
F = [x.^2 y.^2 x.*y x y ones(m,1)];
end