function [mu, var, tts, pts]=HilbertSpace(x,y,x_s,uL,lL,sigma_y,N_m,sigma_SE,l_SE)

P=Lambda(N_m,sigma_SE,l_SE,uL,lL);
w=zeros(N_m,1);

%Learning
N=size(x,2);
training_times=zeros(N,1);
for t=1:N
    tic;
    %Perform the information matrix update
    phi=Phi(x(t),N_m,uL,lL);
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
prediction_times=zeros(N_s,1);
for t=1:N_s
    tic;
    phi=Phi(x_s(t),N_m,uL,lL);
    mu(t)=phi*w;
    var(t)=phi*P*phi';
    prediction_times(t)=toc;
end
pts=prediction_times;




end

function Lambda=Lambda(m,sigma_SE,l_SE,uL,lL)
%Eigenvalues for the 1D squared exponential basis functions
Lambda=eye(m);
for j=1:m
    eigv=(pi*j/(uL-lL)).^2;
    Lambda(j,j)=spectral_density_SE(sqrt(eigv),sigma_SE,l_SE);
end
end

function S_SE=spectral_density_SE(omega,sigma_SE,l_SE)
%This function is implemented based on
%Equation 21 in "Modeling and Interpolation of the Ambient Magnetic Field
%by Gaussian Processes" by Manon Kok and Arno Solin, published in
%IEEE TRANSACTIONS ON ROBOTICS, VOL. 34, NO. 4, AUGUST 2018
S_SE=sigma_SE^2*(2*pi*l_SE^2)^(1/2)*exp(-omega^2*l_SE^2/2);

end