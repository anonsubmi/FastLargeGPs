close all; clear;
load('audio_data.mat');

%Define hyperparameters GP and RR-GP
margin=10;
lL=min(xtrain)-margin; %lower bound
uL=max(xtrain)+margin; %upper bound

%Hyperparameters
sigma_SE=0.009;
l_SE=10.895;
sigma_y=0.002;

rs=[6 12 18];
radii=length(rs);


N_u=8000;

for radius=1:radii
    
    r=rs(radius)*l_SE; %Distance limiter for measurements used in LI inference
    
    %Plot the Local approximations
    [mu, var, ~,~,mbar]=LocalInducingInputs(xtrain',ytrain',xtest',uL,lL,sigma_y,N_u,sigma_SE,l_SE,r);
    SMAE=mean(abs(ytest-mu))./mean(abs(ytest));
    MSLL=1./2*mean((mu-ytest).^2./var+log(2*pi*var));
    disp(['Local Inducing points, 100 percent, r=',num2str(rs(radius)),'l_SE, SMAE=',num2str(SMAE),', MSLL=',num2str(MSLL)]);
end

%% Global inference section

[muIndG, varIndG]=InducingInputs(xtrain',ytrain',xtest',uL,lL,sigma_y,N_u,sigma_SE,l_SE);
SMSEIndG=mean(abs(ytest-muIndG))./mean(abs(ytest));
disp(['Inducing points, 100 percent, SMAE=',num2str(SMSEIndG)]);
[muSKIG, varSKIG]=SKI(xtrain',ytrain',xtest',uL,lL,sigma_y,N_u,sigma_SE,l_SE,true);
SMSESKIG=mean(abs(ytest-muSKIG))./mean(abs(ytest));
disp(['SKI, 100 percent, SMAE=',num2str(SMSESKIG)]);
[muHSG, varHSG]=HilbertSpace(xtrain',ytrain',xtest',uL,lL,sigma_y,N_u,sigma_SE,l_SE);
SMSEHilbertG=mean(abs(ytest-muHSG))./mean(abs(ytest));
disp(['Hilbert Space, 100 percent, SMAE=',num2str(SMSEHilbertG)]);

%% Display MSLL scores

MSLLIndG=1./2*mean((muIndG-ytest).^2./varIndG+log(2*pi*varIndG));
disp(['Inducing points, 100 percent, MSLL=',num2str(MSLLIndG)]);
MSLLSKIG=1./2*mean((muSKIG-ytest).^2./varSKIG+log(2*pi*varSKIG));
disp(['SKI, 100 percent, MSLL=',num2str(MSLLSKIG)]);
MSLLHilbertG=1./2*mean((muHSG-ytest).^2./varHSG+log(2*pi*varHSG));
disp(['Hilbert Space, 100 percent, MSLL=',num2str(MSLLHilbertG)]);

%% Global inference approaches 10 percent

%Now for 10% of the domain
keep_frac=0.1; %Number between 0 and 1
limit=keep_frac*(max(xtrain)-min(xtrain))+min(xtrain);

I_keep=xtrain<limit;
xtrainS=xtrain(I_keep);
ytrainS=ytrain(I_keep);
I_keep=xtest<limit;
xtestS=xtest(I_keep);
ytestS=ytest(I_keep);
%Find the domain borders
margin=1;
lLhere=min(xtrainS)-margin;
uLhere=max(xtrainS)+margin;

N_u=800;

[muInd, varInd]=InducingInputs(xtrainS',ytrainS',xtestS',uLhere,lLhere,sigma_y,N_u,sigma_SE,l_SE);
SMSEInd=mean(abs(ytestS-muInd))./mean(abs(ytestS));
disp(['Inducing points, 10 percent, SMAE=',num2str(SMSEInd)]);
[muSKI, varSKI]=SKI(xtrainS',ytrainS',xtestS',uLhere,lLhere,sigma_y,N_u,sigma_SE,l_SE,true);
SMSESKI=mean(abs(ytestS-muSKI))./mean(abs(ytestS));
disp(['SKI, 10 percent, SMAE=',num2str(SMSESKI)]);
[muHS, varHS]=HilbertSpace(xtrainS',ytrainS',xtestS',uLhere,lLhere,sigma_y,N_u,sigma_SE,l_SE);
SMSEHilbert=mean(abs(ytestS-muHS))./mean(abs(ytestS));
disp(['Hilbert Space, 10 percent, SMAE=',num2str(SMSEHilbert)]);

MSLLInd=1./2*mean((muInd-ytestS).^2./varInd+log(2*pi*varInd));
disp(['Inducing points, 10 percent, MSLL=',num2str(MSLLInd)]);
MSLLSKI=1./2*mean((muSKI-ytestS).^2./varSKI+log(2*pi*varSKI));
disp(['SKI, 10 percent, MSLL=',num2str(MSLLSKI)]);
MSLLHilbert=1./2*mean((muHS-ytestS).^2./varHS+log(2*pi*varHS));
disp(['Hilbert Space, 10 percent, MSLL=',num2str(MSLLHilbert)]);

%% Local domain 10 percent

for radius=1:radii
    
    r=rs(radius)*l_SE; %Distance limiter for measurements used in LI inference
    
    N_u=800;
    
    %Plot the Local approximations
    [mu, var, ~,~,mbar]=LocalInducingInputs(xtrainS',ytrainS',xtestS',uLhere,lLhere,sigma_y,N_u,sigma_SE,l_SE,r);
    SMAE=mean(abs(ytestS-mu))./mean(abs(ytestS));
    MSLL=1./2*mean((mu-ytestS).^2./var+log(2*pi*var));
    disp(['Local Inducing points, 10 percent, r=',num2str(rs(radius)),'l_SE, SMAE=',num2str(SMAE),', MSLL=',num2str(MSLL)]);
end



%% Saving results

save('ComputeTableValues.mat');
