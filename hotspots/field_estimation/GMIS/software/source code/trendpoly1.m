function F = trendpoly1(x,y)
%����Ϊ����
%����ֵ��
%       xΪ������
%       yΪ������
%���ֵ��
%       FΪ���ƾ���

n = length(x);
if length(x) ~= length(y)
    errordlg('��������ά��Ӧ�����');
    return;
end

F = [x y ones(n,1)];
end