clear; close all;
load('seadepth.mat');

seed=42;
rng(seed);

%Define hyperparameters GP and RR-GP
Omega=[1 1; size(seadepth)]'; %Domain

%Hyperparameters
sigma_SE=std(seadepth(:));
l_SE=[4 4];
sigma_y=0.1*sigma_SE;

%Distance limiters
fontsize=14;
%Number of inducing inputs along dim 1
N_u(1)=ceil(1*(Omega(1,2)-Omega(1,1))./l_SE(1)); 
%Number of inducing inputs along dim 2
N_u(2)=ceil(1*(Omega(2,2)-Omega(2,1))./l_SE(2)); 
m=N_u(1)*N_u(2);
r=3*l_SE; %Radial basis function truncation

%Find x-locations
x1=Omega(1,1):Omega(1,2);
x2=Omega(2,1):Omega(2,2);
[X2,X1]=meshgrid(x2, x1);
x=[X1(:)'; X2(:)'];

N=0.01*size(seadepth(:),1);

%Draw training and test samples
sz=size(seadepth(:),1);
indices=randi(sz,N,1);
indices_meas=indices(1:N);

%Find the coordinates of the inputs
x_meas=x(:,indices_meas);

%Make a low-res grid for
res=10;
x1low=Omega(1,1):res:Omega(1,2);
x2low=Omega(2,1):res:Omega(2,2);
[X2low,X1low]=meshgrid(x2low, x1low);
xlow=[X1low(:)'; X2low(:)'];
ylow=seadepth(x1low,x2low);
ylow=ylow(:);

%Find the labels connected with the inputs
y_meas=seadepth(indices_meas);

%Normalise the labels
avg_depth=mean(seadepth(:));
y_meas=y_meas-avg_depth;

[mu, variance, training_time, prediction_time]=FloatingDomainGP2D(x_meas,y_meas,xlow,Omega,N_u,sigma_SE,l_SE,r,sigma_y);
mu=mu+avg_depth;

save('WorkspaceProcessDataFewerMeasurements.mat');

figure; clf;
Opacity=-reshape(variance,size(X1low));
surf(reshape(mu,size(X1low)),'AlphaData',Opacity,'FaceAlpha','flat','EdgeColor','none');
alim([-sigma_SE^2-sigma_y^2 0]);
view(2);
colormap(viridis());
grid off;
box off;
axis off;
saveas(gca,'BathymetryPredictionsFewerMeasurements.fig');

A=viridis();
%Normalise Y data
mu_normalised=reshape(mu,size(X1low))-min(min(reshape(mu,size(X1low))));
mu_normalised=mu_normalised./max(max(mu_normalised));
indices=ceil((mu_normalised).*254+1);

%Normalise Opacity
variance_normalized=reshape(variance-min(min(variance)),size(X1low));
variance_normalized=variance_normalized./max(max(variance_normalized));

%Quick calculation
ColorPicture=reshape(A(indices,:),size(mu_normalised,1),size(mu_normalised,2),3);
ColorPicture=flip(ColorPicture,1);
variance_normalized=flip(variance_normalized,1);
imagename='BathymetryPredictionsFewerMeasurements';
imwrite(ColorPicture,[imagename,'.png'],'Alpha',variance_normalized);
