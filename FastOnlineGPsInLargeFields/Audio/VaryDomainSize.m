close all; clear;
load('audio_data.mat');

%Define hyperparameters GP and RR-GP
lL=min(xfull); %lower bound
uL=max(xfull); %upper bound

%Hyperparameters 
sigma_SE=0.009;
l_SE=10.895;
sigma_y=0.002;

%Settings for experimental results
experiments=10;
domain_fractions=0.1:0.1:1;
params=length(domain_fractions);
N_us=round(linspace(2000,9000,experiments));

rs=[6 12 18];
radii=length(rs);
ms=zeros(params,1);
domain_percentages=round(domain_fractions*100);
TTsInd=zeros(experiments,params);
PTsInd=zeros(experiments,params);
TTsSKI=zeros(experiments,params);
PTsSKI=zeros(experiments,params);
TTsHS=zeros(experiments,params);
PTsHS=zeros(experiments,params);
TTsLoc=zeros(experiments,params,radii);
PTsLoc=zeros(experiments,params,radii);


for param=1:params
    
    load('audio_data.mat');
    
    keep_frac=domain_fractions(param); %Number between 0 and 1
    limit=keep_frac*(uL-lL)+lL;
    
    I_keep=xfull<limit;
    xfull=xfull(I_keep);
    yfull=yfull(I_keep);
    
    %Find the domain borders
    margin=1000;
    lLhere=min(xfull)-margin;
    uLhere=max(xfull)+margin;
    
    %Then, pick a subset of the measurements, as our only goal is to
    %time the inclusion time for one new measurement
    idRand = randperm(length(yfull));
    I_keep=idRand(1:experiments);
    xfull=xfull(I_keep);
    yfull=yfull(I_keep);
    
    N_u=ceil(1.4526*(uLhere-lLhere)./l_SE); %Number of inducing inputs along dim 1
    ms(param)=N_u;
    
    %Sample some random points to make a prediction in
    margin=2000;
    xpred=(uLhere-lLhere-2*margin)*rand(experiments,1)+lLhere+margin;
    
        %Compute the inducing point prediction
    r=50*l_SE;
    if ms(param)<35000
        [~, ~, tts, pts]=InducingInputs(xfull',yfull',xpred',uLhere,lLhere,sigma_y,N_u,sigma_SE,l_SE);                                                                                              
        TTsInd(:,param)=tts;
        PTsInd(:,param)=pts;
        [~, ~,tts, pts]=SKI(xfull',yfull',xpred',uLhere,lLhere,sigma_y,N_u,sigma_SE,l_SE,false);                                                                                              
        TTsSKI(:,param)=tts;
        PTsSKI(:,param)=pts;
        [~, ~, tts, pts]=HilbertSpace(xfull',yfull',xpred',uLhere,lLhere,sigma_y,N_u,sigma_SE,l_SE);                                                                                              
        TTsHS(:,param)=tts;
        PTsHS(:,param)=pts;
    else
        TTsInd(:,param)=NaN;
        PTsInd(:,param)=NaN;
        TTsSKI(:,param)=NaN;
        PTsSKI(:,param)=NaN;
        TTsHS(:,param)=NaN;
        PTsHS(:,param)=NaN;
    end
    
    for radius=1:radii
        
        r=rs(radius)*l_SE; %Distance limiter for measurements used in LI inference
        
        %Plot the Local approximations
        [mu, var, tts, pts]=LocalInducingInputs(xfull',yfull',xpred',uLhere,lLhere,sigma_y,N_u,sigma_SE,l_SE,r);
        TTsLoc(:,param,radius)=tts;
        PTsLoc(:,param,radius)=pts;
        
        
        
    end
    
end

save('VaryDomainSize.mat');
