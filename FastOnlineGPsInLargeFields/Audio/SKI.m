function [mu, var, tts, pts]=SKI(x,y,x_s,uL,lL,sigma_y,N_u,sigma_SE,l_SE,exact)

x_u=linspace(lL,uL,N_u); %Inducing input locations
l_u=x_u(2)-x_u(1);
%Pre-processing step
Kuu=Kern(x_u,x_u,sigma_SE,l_SE);
I=zeros(N_u,N_u);
iota=zeros(N_u,1);

%Learning
N=size(x,2);
training_times=zeros(N,1);
for t=1:N
    tic;
    %Perform the information matrix update
    ls1=max(floor((x(t)-lL)./l_u+1),1);
    us1=min(ceil((x(t)-lL)./l_u+1),N_u);
    set=ls1:us1;
    phi=Hat(x(t),x_u(set),l_u);
    H=phi;
    I(set,set)=I(set,set)+1./sigma_y^2*(H'*H);
    iota(set)=iota(set)+1./sigma_y^2*H'*y(t);
    training_times(t)=toc;
end
tts=training_times;
I=sparse(I);

%Prediction
N_s=size(x_s,2);
mu=zeros(N_s,1);
var=zeros(N_s,1);
max_set_length=0;
prediction_times=zeros(N_s,1);
tol=1e-6;
maxiter=1;
if exact
    I=I+inv(Kuu);
    temp=inv(I);
end

for t=1:N_s
    tic;
    phi=Hat(x_s(t),x_u,l_u);
    if exact
       mu(t)=phi*(temp*iota);
       var(t)=phi*(temp*phi'); 
    else
        mu(t)=phi*(cgs(I,iota,tol,maxiter));
        var(t)=phi*(cgs(I,phi',tol,maxiter));
    end
    prediction_times(t)=toc;
    max_set_length=max(max_set_length,N_u);
end
pts=prediction_times;




end

function K=Hat(x,y,width)
rows=size(x,2);
columns=size(y,2);
K=zeros(rows,columns);
for i=1:rows
    for j=1:columns
        K(i,j)=(1-abs(x(:,i)-y(:,j))./width).*(abs(x(:,i)-y(:,j))<=width);
    end
end
end