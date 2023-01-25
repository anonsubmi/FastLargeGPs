clear; close all;
load('precip_data.mat');

%Reduce to a subdomain
I_keep=(X(:,3)==X(1,3));
X=X(I_keep,1:3);
y=y(I_keep);

%Center the data
y_mean=mean(y);
y=y-y_mean;

%Find the domain borders
margin=4;
Omega(1,1)=min(X(:,1))-margin;
Omega(1,2)=max(X(:,1))+margin;
Omega(2,1)=min(X(:,2))-margin;
Omega(2,2)=max(X(:,2))+margin;
Omega(3,1)=min(X(:,3))-margin;
Omega(3,2)=max(X(:,3))+margin;

%Hyperparameters for Gaussian process prior
sigma_SE=3.99;
l_SE=[3.094, 2.030, 0.189];
sigma_y=2.789;

%Set fontsize and colours for plots
white=[1 1 1];
black=[0 0 0];
gray=[0.5 0.5 0.5];

densities=[0.5 0.75 1.0 1.25 1.5 1.75];
params=length(densities);
experiments=100;
rs=[3 6 12];
radii=length(rs);
KLDivergencesInd=zeros(experiments,params);
KLDivergences=zeros(experiments,params,radii);
ms=zeros(params,1);
mbars=zeros(params,radii); %In percentages

for param=1:params

density=densities(param);

%Place the inducing points
N_u(1)=ceil(density*(Omega(1,2)-Omega(1,1))./l_SE(1)); %Number of inducing inputs along dim 1
N_u(2)=ceil(density*(Omega(2,2)-Omega(2,1))./l_SE(2)); %Number of inducing inputs along dim 2
N_u(3)=3;
ms(param)=N_u(1)*N_u(2)*N_u(3);

%Sample some random points to make a prediction in
x1=(Omega(1,2)-Omega(1,1)-2*margin)*rand(experiments,1)+Omega(1,1)+margin;
x2=(Omega(2,2)-Omega(2,1)-2*margin)*rand(experiments,1)+Omega(2,1)+margin;
x3=X(1,3)+zeros(experiments,1);
x=[x1 x2 x3]';

%Compute the GP prediction
[muGP, varGP]=GaussianProcess(X',y',x,sigma_y,sigma_SE,l_SE);

%Compute the inducing point prediction
r=50*l_SE;
[muInd, varInd, ~,~, max_set_length]=FloatingDomainGP3D(X',y',x,Omega,N_u,sigma_SE,l_SE,r,sigma_y);

for experiment=1:experiments
    var1=varGP(experiment);
    var2=varInd(experiment);
    mu1=muGP(experiment);
    mu2=muInd(experiment);
    sigma1=sqrt(var1);
    sigma2=sqrt(var2);
    term1=log(sigma2./sigma1);
    term2=(var1+(mu1-mu2).^2)./(2*var2);
    KLDivergencesInd(experiment,param)=term1+term2-1/2;
end

for radius=1:radii

r=rs(radius)*l_SE; %Distance limiter for measurements used in LI inference

%Plot the Local approximations
[mu, var, ~,~, max_set_length]=FloatingDomainGP3D(X',y',x,Omega,N_u,sigma_SE,l_SE,r,sigma_y);
mbars(param,radius)=max_set_length;

for experiment=1:experiments
    var1=varGP(experiment);
    var2=var(experiment);
    mu1=muGP(experiment);
    mu2=mu(experiment);
    sigma1=sqrt(var1);
    sigma2=sqrt(var2);
    term1=log(sigma2./sigma1);
    term2=(var1+(mu1-mu2).^2)./(2*var2);
    KLDivergences(experiment,param,radius)=term1+term2-1/2;
end


end

end

save('EstimateDailyVaryDensities.mat');

%% Plots

fontsize=15;
figure; clf;
legends={};
for radius=1:radii
 hsv=[165/360,1,1-(rs(radius)/18)];
 rgb=hsv2rgb(hsv);
 errorbar(densities,mean(KLDivergences(:,:,radius)),std(KLDivergences(:,:,radius))./5,'Color',rgb,'linewidth',1.3);
 legends{radius}=['$r=',num2str(rs(radius)),'l_{SE}$'];
 hold on;
end
errorbar(densities,mean(KLDivergencesInd),std(KLDivergencesInd)./5,'k','linewidth',1.3);
legends{radius+1}=['Inducing inputs,','$\:r=\infty$'];
set(gca,'YScale','log');
grid on;
box off;
xlabel({'Basis function density,'; '$m$ per lengthscale per dimension'},'Interpreter','Latex');
ylabel('KLD true and approximate posterior','Interpreter','Latex');
legend(legends,'Interpreter','Latex','location','southwest');
set(gca, 'FontName', 'Times');
set(gca,'fontsize',fontsize);
exportgraphics(gca,'PrecipKLConvergence.png','Resolution',500);

fontsize=15;
figure; clf;
legends={};
for radius=1:radii
 hsv=[165/360,1,1-(rs(radius)/18)];
 rgb=hsv2rgb(hsv);
 errorbar(mbars(:,radius),mean(KLDivergences(:,:,radius)),std(KLDivergences(:,:,radius))./5,'Color',rgb,'linewidth',1.3);
 legends{radius}=['$r=',num2str(rs(radius)),'l_{SE}$'];
 hold on;
end
errorbar(ms,mean(KLDivergencesInd),std(KLDivergencesInd)./5,'k','linewidth',1.3);
legends{radius+1}=['Inducing inputs,','$\:r=\infty$'];
set(gca,'YScale','log');
set(gca,'XScale','log');
grid on;
box off;
xlabel({'Number of basis functions'; '$m$ per lengthscale per dimension'},'Interpreter','Latex');
ylabel('KLD true and approximate posterior','Interpreter','Latex');
legend(legends,'Interpreter','Latex','location','southwest');
set(gca, 'FontName', 'Times');
set(gca,'fontsize',fontsize);
exportgraphics(gca,'PrecipKLComparison.png','Resolution',500);