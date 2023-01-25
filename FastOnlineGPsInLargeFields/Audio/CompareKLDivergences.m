close all; clear;
load('audio_data.mat');
rng(42);

%Define hyperparameters GP and RR-GP
lL=min(xfull); %lower bound
uL=max(xfull); %upper bound

%Hyperparameters
sigma_SE=0.009;
l_SE=10.895;
sigma_y=0.002;

%Settings for experimental results
experiments=100;
densities=[0.5 0.8 1.1 1.4 1.7 1.9];
params=length(densities);

rs=[6 12 18];
radii=length(rs);
ms=zeros(params,1);
mbars=zeros(params,radii);
KLDivergencesInd=zeros(experiments,params);
KLDivergencesSKI=zeros(experiments,params);
KLDivergencesHilbert=zeros(experiments,params);
KLDivergences=zeros(experiments,params,radii);

keep_frac=0.1; %Number between 0 and 1
limit=keep_frac*(uL-lL)+lL;

I_keep=xfull<limit;
xfull=xfull(I_keep);
yfull=yfull(I_keep);

%Find the domain borders
margin=50;
lLhere=min(xfull)-margin;
uLhere=max(xfull)+margin;

%Then, pick a subset of the measurements
idRand = randperm(length(yfull));
I_keep=idRand(1:experiments);
xfull=xfull(I_keep);
yfull=yfull(I_keep);

%Sample some random points to make a prediction in
xpred=(uLhere-lLhere-2*margin)*rand(experiments,1)+lLhere+margin;

%Compute the GP prediction
[muGP, varGP]=GaussianProcess(xfull',yfull',xpred',sigma_y,sigma_SE,l_SE);

for param=1:params
    
    density=densities(param);
    N_u=ceil(density*(uLhere-lLhere)./l_SE); %Number of inducing inputs along dim 1
    ms(param)=N_u;
    
    %Compute the inducing point prediction
    r=50*l_SE;
    if ms(param)<6000
        [muInd, varInd]=InducingInputs(xfull',yfull',xpred',uLhere,lLhere,sigma_y,N_u,sigma_SE,l_SE);
        for experiment=1:experiments
            var1=varInd(experiment);
            var2=varGP(experiment);
            mu1=muInd(experiment);
            mu2=muGP(experiment);
            sigma1=sqrt(var1);
            sigma2=sqrt(var2);
            term1=log(sigma2./sigma1);
            term2=(var1+(mu1-mu2).^2)./(2*var2);
            KLDivergencesInd(experiment,param)=term1+term2-1/2;
        end
        [muSKI, varSKI]=SKI(xfull',yfull',xpred',uLhere,lLhere,sigma_y,N_u,sigma_SE,l_SE,true);
        for experiment=1:experiments
            var1=varSKI(experiment);
            var2=varGP(experiment);
            mu1=muSKI(experiment);
            mu2=muGP(experiment);
            sigma1=sqrt(var1);
            sigma2=sqrt(var2);
            term1=log(sigma2./sigma1);
            term2=(var1+(mu1-mu2).^2)./(2*var2);
            KLDivergencesSKI(experiment,param)=term1+term2-1/2;
        end
        [muHS, varHS]=HilbertSpace(xfull',yfull',xpred',uLhere,lLhere,sigma_y,N_u,sigma_SE,l_SE);
        for experiment=1:experiments
            var1=varHS(experiment);
            var2=varGP(experiment);
            mu1=muHS(experiment);
            mu2=muGP(experiment);
            sigma1=sqrt(var1);
            sigma2=sqrt(var2);
            term1=log(sigma2./sigma1);
            term2=(var1+(mu1-mu2).^2)./(2*var2);
            KLDivergencesHilbert(experiment,param)=term1+term2-1/2;
        end
    else
        for experiment=1:experiments
            KLDivergencesInd(experiment,param)=NaN;
            KLDivergencesSKI(experiment,param)=NaN;
            KLDivergencesHilbert(experiment,param)=NaN;
        end
    end
    
    for radius=1:radii
        
        r=rs(radius)*l_SE; %Distance limiter for measurements used in LI inference
        
        %Plot the Local approximations
        [mu, var, ~,~,mbar]=LocalInducingInputs(xfull',yfull',xpred',uLhere,lLhere,sigma_y,N_u,sigma_SE,l_SE,r);
        mbars(param,radius)=mbar;
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

save('CompareConvergence.mat');

%% Plot results all

fontsize=15;
figure; clf;
legends={};
for radius=1:radii
    hsv=[165/360,1,1-((rs(radius))/40)];
    rgb=hsv2rgb(hsv);
    [neg_devs, pos_devs]=negposdevs(KLDivergences(:,:,radius));
    errorbar(densities,mean(KLDivergences(:,:,radius)),neg_devs,pos_devs,'Color',rgb,'linewidth',1.3);
    legends{radius}=['$r=',num2str(rs(radius)),'l_{\mathrm{SE}}$'];
    hold on;
end
[neg_devs, pos_devs]=negposdevs(KLDivergencesInd);
errorbar(densities,mean(KLDivergencesInd),neg_devs,pos_devs,'k','linewidth',1.3);
legends{radius+1}=['Inducing inputs,','$\:r=\infty$'];
set(gca,'YScale','log');
grid on;
box off;
xlabel({'Basis function density'},'Interpreter','Latex');
ylabel('KLD true and approximate posterior','Interpreter','Latex');
legend(legends,'Interpreter','Latex','location','southwest');
set(gca, 'FontName', 'Times');
set(gca,'fontsize',fontsize);
exportgraphics(gca,'AudioConvergence.png','Resolution',500);

% Plot results KL Compare

fontsize=15;
figure; clf;
legends={};
k=0;
for radius=1:radii
    hsv=[165/360,1,1-((rs(radius))/40)];
    rgb=hsv2rgb(hsv);
    [neg_devs, pos_devs]=negposdevs(KLDivergences(:,:,radius));
    errorbar(mbars(:,radius),mean(KLDivergences(:,:,radius)),neg_devs,pos_devs,'Color',rgb,'linewidth',1.3);
    k=k+1; legends{k}=['$r=',num2str(rs(radius)),'l_{\mathrm{SE}}$'];
    hold on;
end
[neg_devs, pos_devs]=negposdevs(KLDivergencesInd);
errorbar(ms,mean(KLDivergencesInd),neg_devs,pos_devs,'k','linewidth',1.3);
k=k+1; legends{k}=['Inducing inputs'];
[neg_devs, pos_devs]=negposdevs(KLDivergencesHilbert);
errorbar(ms,mean(KLDivergencesHilbert),neg_devs,pos_devs,'m--','linewidth',1.3);
k=k+1; legends{k}='Hilbert Space';
[neg_devs, pos_devs]=negposdevs(KLDivergencesSKI);
errorbar(ms,mean(KLDivergencesSKI),neg_devs,pos_devs,'r:','linewidth',1.3);
k=k+1; legends{k}='SKI';
set(gca,'YScale','log');
set(gca,'XScale','log');
xlims=xlim();
xlim([0.5 xlims(2)]);
grid on;
box off;
xlabel({'Number of basis functions'},'Interpreter','Latex');
ylabel('KLD true and approximate posterior','Interpreter','Latex');
legend(legends,'Interpreter','Latex','location','southwest');
set(gca, 'FontName', 'Times');
set(gca,'fontsize',fontsize);
exportgraphics(gca,'AudioKLCompare.png','Resolution',500);