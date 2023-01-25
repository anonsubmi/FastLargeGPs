clear; close all;
load('precip_data.mat');

%Center the data
y_mean=mean(y);
y=y-y_mean;

%Find the total domain size
xmax=max(X(:,1));
xmin=min(X(:,1));
ymax=max(X(:,2));
ymin=min(X(:,2));
zmax=max(X(:,3));
zmin=min(X(:,3));
xc=(xmax+xmin)./2;
yc=(ymax+ymin)./2;
zc=(zmax+zmin)./2;

Omega(3,1)=min(X(:,3));
Omega(3,2)=max(X(:,3));

%Hyperparameters for Gaussian process prior
sigma_SE=3.99;
l_SE=[3.094, 2.030, 0.189];
sigma_y=2.789;

%Set fontsize and colours for plots
white=[1 1 1];
black=[0 0 0];
gray=[0.5 0.5 0.5];

density=1;
domain_fractions=[0.025 0.05 0.075 0.1 0.125 0.15 0.2];
params=length(domain_fractions);
experiments=100;
rs=[3 6 12];
radii=length(rs);
ms=zeros(params,1);
KLDivergencesInd=zeros(experiments,params);
KLDivergences=zeros(experiments,params,radii);
domain_percentages=round(domain_fractions*100);
TTsInd=zeros(experiments,params);
PTsInd=zeros(experiments,params);
TTsLoc=zeros(experiments,params,radii);
PTsLoc=zeros(experiments,params,radii);

for param=1:params
    
    load('precip_data.mat');
    y=y-y_mean;
    
    keep_frac=domain_fractions(param); %Number between 0 and 1
    xwidth=1*(xmax-xmin);
    ywidth=1*(ymax-ymin);
    DomainSizes(param)=xwidth*ywidth;
    zwidth=keep_frac*(zmax-zmin);
    ylower=yc-ywidth./2;
    yupper=yc+ywidth./2;
    xlower=xc-xwidth./2;
    xupper=xc+xwidth./2;
    zlower=zc-zwidth./2;
    zupper=zc+zwidth./2;
    
    I_keep=(xlower<(X(:,1))) & ((X(:,1))<xupper) & ((X(:,2))<yupper) & (ylower<(X(:,2))) & ((X(:,3))<zupper) & (zlower<(X(:,3)));
    X=X(I_keep,:);
    y=y(I_keep);
    
    %Find the domain borders
    margin=0.5;
    Omega(1,1)=min(X(:,1))-margin;
    Omega(1,2)=max(X(:,1))+margin;
    Omega(2,1)=min(X(:,2))-margin;
    Omega(2,2)=max(X(:,2))+margin;
    Omega(3,1)=min(X(:,3));
    Omega(3,2)=max(X(:,3));
    
    %Then, pick a subset of the measurements, as our only goal is to
    %time the inclusion time for one new measurement
    idRand = randperm(length(y));
    IdKeep=idRand(1:experiments);
    X=X(IdKeep,:);
    y=y(IdKeep);
    
    %Place the inducing points
    N_u(3)=Omega(3,2)-Omega(3,1); %Number of inducing points along dim 3 - equivalent to number of days?
    N_u(1)=ceil(density*(Omega(1,2)-Omega(1,1))./l_SE(1)); %Number of inducing inputs along dim 1
    N_u(2)=ceil(density*(Omega(2,2)-Omega(2,1))./l_SE(2)); %Number of inducing inputs along dim 2
    ms(param)=N_u(1)*N_u(2)*N_u(3);
    
    %Find the full grid of inducing points
    x1_u=linspace(Omega(1,1),Omega(1,2),N_u(1)); %Inducing inputs along first dimension
    x2_u=linspace(Omega(2,1),Omega(2,2),N_u(2)); %Inducing inputs along second dimension
    [x1_um,x2_um]=meshgrid(x1_u,x2_u);
    x_u=[x1_um(:);x2_um(:)];
    
    %Sample some random points to make a prediction in
    x1=(Omega(1,2)-Omega(1,1)-2*margin)*rand(experiments,1)+Omega(1,1)+margin;
    x2=(Omega(2,2)-Omega(2,1)-2*margin)*rand(experiments,1)+Omega(2,1)+margin;
    x3=(Omega(3,2)-Omega(3,1))*rand(experiments,1)+Omega(3,1);
    x=[x1 x2 x3]';
    
    %Compute the inducing point prediction
    r=50*l_SE;
    if ms(param)<35000
        [muInd, varInd, tts, pts]=InducingPointGP3DKF(X',y',x,Omega,N_u,sigma_SE,l_SE,r,sigma_y);
        
        TTsInd(:,param)=tts;
        PTsInd(:,param)=pts;
    else
        TTsInd(:,param)=NaN;
        PTsInd(:,param)=NaN;
    end
    
    for radius=1:radii
        
        r=rs(radius)*l_SE; %Distance limiter for measurements used in LI inference
        
        %Plot the Local approximations
        [mu, var, tts, pts]=FloatingDomainGP3D(X',y',x,Omega,N_u,sigma_SE,l_SE,r,sigma_y);
        TTsLoc(:,param,radius)=tts;
        PTsLoc(:,param,radius)=pts;
        
    end
    
end

%% Plots

%Training time plots

fontsize=15;
figure; clf;
legends={};
for radius=1:radii
    hsv=[165/360,1,1-(rs(radius)/40)];
    rgb=hsv2rgb(hsv);
    [neg_devs, pos_devs]=negposdevs(TTsLoc(:,:,radius));
    errorbar(domain_fractions,mean(TTsLoc(:,:,radius),1),neg_devs, pos_devs,'Color',rgb,'linewidth',1.3);
    legends{radius}=['$r=',num2str(rs(radius)),'l_{SE}$'];
    hold on;
end
[neg_devs, pos_devs]=negposdevs(TTsInd);
errorbar(domain_fractions,mean(TTsInd),neg_devs, pos_devs,'k','linewidth',1.3);
legends{radius+1}=['Inducing inputs,','$\:r=\infty$'];
set(gca,'YScale','log');
grid on;
box off;
xlabel({'Domain Size'},'Interpreter','Latex');
ylabel('Training Time','Interpreter','Latex');
legend(legends,'Interpreter','Latex','location','southwest');
set(gca, 'FontName', 'Times');
set(gca,'fontsize',fontsize);

fontsize=15;
figure; clf;
legends={};
for radius=1:radii
    hsv=[165/360,1,1-(rs(radius)/40)];
    rgb=hsv2rgb(hsv);
    [neg_devs, pos_devs]=negposdevs(PTsLoc(:,:,radius));
    errorbar(domain_fractions,mean(PTsLoc(:,:,radius),1),neg_devs, pos_devs,'Color',rgb,'linewidth',1.3);
    legends{radius}=['$r=',num2str(rs(radius)),'l_{SE}$'];
    hold on;
end
[neg_devs, pos_devs]=negposdevs(PTsInd);
errorbar(domain_fractions,mean(PTsInd),neg_devs, pos_devs,'k','linewidth',1.3);
legends{radius+1}=['Inducing inputs'];
set(gca,'YScale','log');
grid on;
box off;
xlabel({'Domain Size'},'Interpreter','Latex');
ylabel('Prediction Time','Interpreter','Latex');
legend(legends,'Interpreter','Latex','location','southwest');
set(gca, 'FontName', 'Times');
set(gca,'fontsize',fontsize);

save('EstimateDailyVaryDomainSizeAndTime.mat');