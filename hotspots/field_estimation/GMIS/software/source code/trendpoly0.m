function F = trendpoly0(x,y)
%����Ϊ����
%����ֵ��
%       xΪ������
%       yΪ������
%���ֵ��
%       FΪ���ƾ���
x = x/10^6;
y = y/10^6;
n = length(x);
if length(x) ~= length(y)
    errordlg('��������ά��Ӧ�����');
    return;
end

F = ones(n,1);
end