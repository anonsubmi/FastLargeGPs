function [mu, var, tts, pts]=InducingInputs(x,y,x_s,uL,lL,sigma_y,N_u,sigma_SE,l_SE)

x_u=linspace(lL,uL,N_u); %Inducing input locations
%Pre-processing step
Kuu=Kern(x_u,x_u,sigma_SE,l_SE);
P=inv(Kuu);
w=zeros(N_u,1);

%Learning
N=size(x,2);
training_times=zeros(N,1);
for t=1:N
    tic;
    %Perform the information matrix update
    phi=Kern(x(t),x_u,sigma_SE,l_SE);
    H=phi;
    S=H*P*H'+sigma_y^2;
    K=P*H'*inv(S);
    P=P-K*S*K';
    z=y(t)-phi*w;
    w=w+K*z;
    training_times(t)=toc;
end
tts=training_times;

%Prediction
N_s=size(x_s,2);
mu=zeros(N_s,1);
var=zeros(N_s,1);
prediction_times=zeros(N_s,1);
for t=1:N_s
    tic;
    phi=Kern(x_s(t),x_u,sigma_SE,l_SE);
    mu(t)=phi*w;
    var(t)=phi*P*phi';
    prediction_times(t)=toc;
end
pts=prediction_times;




end