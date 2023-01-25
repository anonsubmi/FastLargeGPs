function [mu, var, tts, pts]=InducingPointGP3DKF(x,y,x_s,Omega,N_u,sigma_SE,l_SE,r,sigma_y)
x1_u=linspace(Omega(1,1),Omega(1,2),N_u(1)); %Inducing inputs along first dimension
x2_u=linspace(Omega(2,1),Omega(2,2),N_u(2)); %Inducing inputs along second dimension
x3_u=linspace(Omega(3,1),Omega(3,2),N_u(3)); %Inducing inputs along third dimension

%Now, store and refer to bigger structures
M=N_u(1)*N_u(2)*N_u(3);
[xu2_set, xu1_set,xu3_set]=meshgrid(x2_u,x1_u,x3_u);
xu_set=[xu1_set(:) xu2_set(:) xu3_set(:)]';
Kuu=Kern3D(xu_set,xu_set,sigma_SE,l_SE);
P=Kuu;
w=zeros(M,1);

%Learning
N=size(x,2);
training_times=zeros(N,1);
for t=1:N
    tic;
    %Perform the information matrix update
    phi=Kern3D(x(:,t),xu_set,sigma_SE,l_SE);
    H=phi;
    S=H*P*H'+sigma_y^2;
    K=P*H'*inv(S);
    P=P-K*S*K';
    z=phi*w-y(t);
    w=w-K*z;
    training_times(t)=toc;
end
tts=training_times;

%Prediction
N_s=size(x_s,2);
mu=zeros(N_s,1);
var=zeros(N_s,1);
max_set_length=0;
prediction_times=zeros(N_s,1);
for t=1:N_s
    tic;
    phi=Kern3D(x_s(:,t),xu_set,sigma_SE,l_SE);
    Kss=Kern3D(x_s(:,t),x_s(:,t),sigma_SE,l_SE);
    Qss=phi*P*phi';
    mu(t)=phi*w;
    var(t)=sigma_y^2*phi*P*phi'+Kss-sigma_y^2*Qss;
    max_set_length=max(max_set_length,M);
    prediction_times(t)=toc;
end
pts=prediction_times;

end