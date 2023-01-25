function [mu, var, mean_tt, std_tt, mean_pt, std_pt]=InducingInputs(x,y,x_s,Omega,N_u1,N_u2,sigma_SE,l_SE,sigma_y)
%mean_tt measures the mean training time
%std_tt measures the standard deviation of the training time
%mean_pt measures the mean prediction time
%std_pt measures the standard deviation of the prediction time

x1_u=linspace(Omega(1,1),Omega(1,2),N_u1); %Inducing inputs along first dimension
x2_u=linspace(Omega(2,1),Omega(2,2),N_u2); %Inducing inputs along second dimension

[xu1, xu2]=meshgrid(x1_u,x2_u);
xu1=xu1';
xu2=xu2';
xu=[xu1(:) xu2(:)]';

%Now, store and refer to bigger structures
I=zeros(N_u1*N_u2,N_u1*N_u2);
iota=zeros(N_u1*N_u2,1);

%Learning
N=size(x,2);
training_times=zeros(N,1);
for t=1:N
    tic;
    %Perform the information matrix update
    phi=Kern(x(:,t),xu,sigma_SE,l_SE);
    I_mat=phi'*phi;
    iota_mat=phi'*y(t);
    I(:,:)=I(:,:)+I_mat;
    iota(:,:)=iota(:,:)+iota_mat;
    training_times(t)=toc;
end
mean_tt=mean(training_times);
std_tt=std(training_times);

%Prediction
N_s=size(x_s,2);
mu=zeros(N_s,1);
var=zeros(N_s,1);

temp=I+sigma_y^2.*Kern(xu,xu,sigma_SE,l_SE);
weights=(temp\(iota));
P=inv(temp);
prediction_times=zeros(N_s,1);
for t=1:N_s
tic;
phi=Kern(x_s(:,t),xu,sigma_SE,l_SE);
mu(t)=phi*weights;
var(t)=diag(sigma_y^2*phi*(P*(phi')));
prediction_times(t)=toc;
end
mean_pt=mean(prediction_times);
std_pt=std(prediction_times);
end