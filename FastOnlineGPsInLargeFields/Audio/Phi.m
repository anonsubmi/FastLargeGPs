function phi=Phi(x,m,uL,lL)
%Basis functions for the SE kernel function

N=size(x,2);
phi=zeros(N,m);
for i=1:N
    for j=1:m
        phi(i,j)=1/sqrt(0.5*(uL-lL))*sin(pi*j*(x(i)-lL)/(uL-lL));
    end
end

end