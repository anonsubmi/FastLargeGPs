function K=Kern(x,y,sigma_SE,l_SE)
rows=size(x,2);
columns=size(y,2);
K=zeros(rows,columns);
for i=1:rows
    for j=1:columns
        nom=norm((x(:,i)-y(:,j))./l_SE')^2;
        den=2;
        core=nom/den;
        K(i,j)=sigma_SE^2*exp(-core);
    end
end
end