function F = trendpoly3(x,y)
%����Ϊ����
%����ֵ��
%       xΪ������
%       yΪ������
%���ֵ��
%       FΪ���ƾ���
if length(x) ~= length(y)
    errordlg('��������ά��Ӧ�����');
    return;
end
m=length(x);
F = [x.^2 y.^2 x.*y x y ones(m,1)];
end