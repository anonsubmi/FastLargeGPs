clear; close all;
load('EstimateDailyVaryDomainSizeAndTime.mat');

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
legend(legends,'Interpreter','Latex','location','northwest');
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
legend(legends,'Interpreter','Latex','location','northwest');
set(gca, 'FontName', 'Times');
set(gca,'fontsize',fontsize);

fontsize=15;
figure; clf;
legends={};
for radius=1:radii
    hsv=[165/360,1,1-(rs(radius)/40)];
    rgb=hsv2rgb(hsv);
    [neg_devs, pos_devs]=negposdevs(PTsLoc(:,:,radius)+TTsLoc(:,:,radius));
    errorbar(domain_fractions,mean(PTsLoc(:,:,radius)+TTsLoc(:,:,radius),1),neg_devs, pos_devs,'Color',rgb,'linewidth',1.3);
    legends{radius}=['$r=',num2str(rs(radius)),'l_{SE}$'];
    hold on;
end
[neg_devs, pos_devs]=negposdevs(PTsInd+TTsInd);
errorbar(domain_fractions,mean(PTsInd+TTsInd),neg_devs, pos_devs,'k','linewidth',1.3);
legends{radius+1}='Inducing inputs';
set(gca,'YScale','log');
grid on;
box off;
xlabel({'Domain Size'},'Interpreter','Latex');
ylabel('Online inference time','Interpreter','Latex');
legend(legends,'Interpreter','Latex','location','northwest');
set(gca, 'FontName', 'Times');
set(gca,'fontsize',fontsize);
exportgraphics(gca,'PrecipVaryDomain.png','Resolution',500);
