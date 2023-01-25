clear; close all;
load('VaryDomainSize.mat');

fontsize=15;
figure; clf;
legends={};
for radius=1:radii
    hsv=[165/360,1,1-(rs(radius)/40)];
    rgb=hsv2rgb(hsv);
    plot(1,1,'Color',rgb,'linewidth',1.3);
    legends{radius}=['$r=',num2str(rs(radius)),'l_{\mathrm{SE}}$'];
    hold on;
end
plot(1,1,'k','linewidth',1.3);
legends{radius+1}='Inducing inputs';
grid on;
box off;
axis off;
legend(legends,'Interpreter','Latex','location','northoutside');
xlim([-1 1]);
ylim([-1 1]);
set(gca, 'FontName', 'Times');
set(gca,'fontsize',fontsize);


figure; clf;
legends={};
for radius=1:radii
    hsv=[165/360,1,1-(rs(radius)/40)];
    rgb=hsv2rgb(hsv);
    [neg_devs, pos_devs]=negposdevs(TTsLoc(:,:,radius));
    errorbar(domain_fractions*(uL-lL)./l_SE,mean(TTsLoc(:,:,radius)),neg_devs, pos_devs,'Color',rgb,'linewidth',1.3);
    legends{radius}=['$r=',num2str(rs(radius)),'l_{\mathrm{SE}}$'];
    hold on;
end
[neg_devs, pos_devs]=negposdevs(TTsInd);
errorbar(domain_fractions*(uL-lL)./l_SE,mean(TTsInd),neg_devs, pos_devs,'k','linewidth',1.3);
legends{radius+1}=['Inducing inputs'];
Data=TTsHS;
[neg_devs, pos_devs]=negposdevs(Data);
errorbar(domain_fractions*(uL-lL)./l_SE,mean(Data),neg_devs, pos_devs,'b','linewidth',1.3);
legends{radius+2}='Hilbert Space';
Data=TTsSKI;
[neg_devs, pos_devs]=negposdevs(Data);
errorbar(domain_fractions*(uL-lL)./l_SE,mean(Data),neg_devs, pos_devs,'r','linewidth',1.3);
legends{radius+3}='SKI';
set(gca,'YScale','log');
grid on;
box off;
xlabel({'Domain Size/$l_{\mathrm{SE}}$'},'Interpreter','Latex');
ylabel('Training Time','Interpreter','Latex');
%legend(legends,'Interpreter','Latex','location','northeast');
set(gca, 'FontName', 'Times');
set(gca,'fontsize',fontsize);

fontsize=15;
figure; clf;
legends={};
for radius=1:radii
    hsv=[165/360,1,1-(rs(radius)/40)];
    rgb=hsv2rgb(hsv);
    [neg_devs, pos_devs]=negposdevs(PTsLoc(:,:,radius));
    errorbar(domain_fractions*(uL-lL)./l_SE,mean(PTsLoc(:,:,radius)),neg_devs, pos_devs,'Color',rgb,'linewidth',1.3);
    legends{radius}=['$r=',num2str(rs(radius)),'l_{\mathrm{SE}}$'];
    hold on;
end
[neg_devs, pos_devs]=negposdevs(PTsInd);
errorbar(domain_fractions*(uL-lL)./l_SE,mean(PTsInd),neg_devs, pos_devs,'k','linewidth',1.3);
legends{radius+1}='Inducing inputs';
Data=PTsHS;
[neg_devs, pos_devs]=negposdevs(Data);
errorbar(domain_fractions*(uL-lL)./l_SE,mean(Data),neg_devs, pos_devs,'b','linewidth',1.3);
legends{radius+2}='Hilbert Space';
Data=PTsSKI;
[neg_devs, pos_devs]=negposdevs(Data);
errorbar(domain_fractions*(uL-lL)./l_SE,mean(Data),neg_devs, pos_devs,'r','linewidth',1.3);
legends{radius+3}='SKI';
set(gca,'YScale','log');
grid on;
box off;
xlabel({'Domain Size/$l_{\mathrm{SE}}$'},'Interpreter','Latex');
ylabel('Prediction Time','Interpreter','Latex');
%legend(legends,'Interpreter','Latex','location','northeast');
set(gca, 'FontName', 'Times');
set(gca,'fontsize',fontsize);

%Plot the sums of prediction training time and prediction time
fontsize=13;
figure; clf;
legends={};
for radius=1:radii
    hsv=[165/360,1,1-(rs(radius)/40)];
    rgb=hsv2rgb(hsv);
    Data=PTsLoc(:,:,radius)+TTsLoc(:,:,radius);
    [neg_devs, pos_devs]=negposdevs(Data);
    errorbar(domain_fractions*(uL-lL)./l_SE,mean(Data),neg_devs, pos_devs,'Color',rgb,'linewidth',1.5);
    legends{radius}=['$r=',num2str(rs(radius)),'l_{\mathrm{SE}}$'];
    hold on;
end
Data=PTsInd+TTsInd;
[neg_devs, pos_devs]=negposdevs(Data);
errorbar(domain_fractions*(uL-lL)./l_SE,mean(Data),neg_devs, pos_devs,'k','linewidth',2);
legends{radius+1}='Inducing inputs';
Data=PTsHS+TTsHS;
[neg_devs, pos_devs]=negposdevs(Data);
errorbar(domain_fractions*(uL-lL)./l_SE,mean(Data),neg_devs, pos_devs,'m--','linewidth',1.5);
legends{radius+2}='Hilbert Space';
Data=PTsSKI+TTsSKI;
[neg_devs, pos_devs]=negposdevs(Data);
errorbar(domain_fractions*(uL-lL)./l_SE,mean(Data),neg_devs, pos_devs,'r:','linewidth',1.5);
legends{radius+3}='SKI';
set(gca,'YScale','log');
grid on;
box off;
xlabel({'Domain Size/$l_{\mathrm{SE}}$'},'Interpreter','Latex');
ylabel('Online inference time','Interpreter','Latex');
h = legend(legends,'Interpreter','Latex','Location','northoutside','orientation','horizontal', 'NumColumns', 3);
%pos = get(h,'Position');
%posx = 0.6;
%posy = 0.43;
%set(h,'Position',[posx posy pos(3) pos(4)]);
set(gca, 'FontName', 'Times');
set(gca,'fontsize',fontsize);
exportgraphics(gca,'TimesAudio.png','Resolution',500);

%Timing results for table
Data=PTsSKI+TTsSKI;
disp(['Online computation time SKI, m=8000:',num2str(mean(Data(:,end))),'+-',num2str(std(Data(:,end)))]);
Data=PTsHS+TTsHS;
disp(['Online computation time Inducing Points, m=8000:',num2str(mean(Data(:,end))),'+-',num2str(std(Data(:,end)))]);
Data=PTsInd+TTsInd;
disp(['Online computation time Hilbert Space, m=8000:',num2str(mean(Data(:,end))),'+-',num2str(std(Data(:,end)))]);
for radius=1:radii
    Data=PTsLoc(:,end,radius)+TTsLoc(:,end,radius); 
    disp(['Online computation time Local, m=8000:',num2str(mean(Data)),'+-',num2str(std(Data))]);
    
end
Data=PTsSKI+TTsSKI;
disp(['Online computation time SKI, m=800:',num2str(mean(Data(:,2))),'+-',num2str(std(Data(:,2)))]);
Data=PTsHS+TTsHS;
disp(['Online computation time Inducing Points, m=800:',num2str(mean(Data(:,2))),'+-',num2str(std(Data(:,2)))]);
Data=PTsInd+TTsInd;
disp(['Online computation time Hilbert Space, m=800:',num2str(mean(Data(:,2))),'+-',num2str(std(Data(:,2)))]);
for radius=1:radii
    Data=PTsLoc(:,end,radius)+TTsLoc(:,2,radius); 
    disp(['Online computation time Local, m=800:',num2str(mean(Data)),'+-',num2str(std(Data))]);
    
end

