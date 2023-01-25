function [mu, var, tts, pts, max_set_length]=LocalInducingInputs(x,y,x_s,uL,lL,sigma_y,N_u,sigma_SE,l_SE,r)


x_u=linspace(lL,uL,N_u); %Inducing input locations
l_u=x_u(2)-x_u(1); %Distance between two inducing points
N=size(x,2); %Number of measurements
I=zeros(N_u,N_u); %Pre-allocate space for the information matrix
iota=zeros(N_u,1); %Pre-allocate space for the information vector
training_times=zeros(N,1);
for t=1:N
    tic;
    %Calculate the set as a function surrounding the current location
    ls1=max(floor((x(t)-r-lL)./l_u+2),1);
    us1=min(ceil((x(t)+r-lL)./l_u),N_u);
    set=ls1:us1;
    phi=sigma_SE^2.*exp(-(x(t)'-x_u(set)).^2./(2*l_SE^2));
    I(set,set)=I(set,set)+phi'*phi;
    iota(set)=iota(set)+phi'*y(t);
    training_times(t)=toc;
end
tts=training_times;

N_s=size(x_s,2);
mu=zeros(N_s,1);
var=zeros(N_s,1);
max_set_length=0;
prediction_times=zeros(N_s,1);
for i=1:N_s
    tic;
    ls1=max(floor((x_s(i)-r./2-lL)./l_u+2),1);
    us1=min(ceil((x_s(i)+r./2-lL)./l_u),N_u);
    set=ls1:us1;
    max_set_length=max(max_set_length,length(set));
    phi=Kern(x_s(i),x_u(set),sigma_SE,l_SE);
    temp=I(set,set)+sigma_y^2.*Kern(x_u(set),x_u(set),sigma_SE,l_SE);
    temp=inv(temp);
    mu(i)=phi*temp*iota(set);
    var(i)=sigma_y^2*phi*temp*phi';
    prediction_times(i)=toc;
end
pts=prediction_times;

end