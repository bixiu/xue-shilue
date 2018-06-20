function [ys,g]=O_Neuron(w,Theta,b,y)
be=[b,-1];
we=[w;Theta];
ys=1/(1+exp(-be*we));
g=ys*(1-ys)*(y-ys);