function [e,b]=H_Neuron(x,v,Gamma,w,g)
%wΪ����Ԫ�����Ȩ����
%vΪ����Ԫ������Ȩ����
%gΪ��һ�㣨����㣩���ݶȾ���
xe=[x,-1];
ve=[v;Gamma];
b=1/(1+exp(-xe*ve));
if size(g,1)==1
    g=g';
end
e=b*(1-b)*(w*g);