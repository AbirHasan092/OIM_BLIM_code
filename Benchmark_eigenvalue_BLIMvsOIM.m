clc;
close all;
clear all;
%parpool;
J = readmatrix('D:\Abir\BLIM\Graphs\G201.txt');
N = length(J);
edge = 0.5*sum(sum(-J));

dec = [0:2^N-1]';
bin=( decimalToBinaryVector(dec));
S = 2*bin-ones(2^N,N);
Hlist=zeros(2^N,1);
for i=1:2^N
    Hlist(i)=-0.5*S(i,:)*J*S(i,:)';
end
Jacobian = zeros(N,N);
eigminlist = zeros(2^N,1);

K=2;
tau =[1];
tauc = [4 12 24];

eigmaxlist = zeros(2^N,length(tauc),length(K));
 m = 1:length(tau)
for l = 1:length(tauc)
for k =1:2^N
    parfor i = 1:N
        for j = 1:N
             if i==j
                Jacobian(i,j)=((K^2/tau(m))*((sech(K*tanh(K*S(k,i))))^2)*(sech(K*S(k,i)))^2)-(sum(-J(i,:))*(1/tauc(l)))-1/tau(m);
               %Jacobian(i,j)=-(sum(-J(i,:))*(1/tauc(l)))-1/tau(m);
            else
               % Jacobian(i,j) = (-K/tauc(l))*(sech(K*S(k,j))^2)*(-J(i,j));
               Jacobian(i,j) = -(-J(i,j))/tauc(l);
             end
        end
    end
    [V,D]=eig(Jacobian);
    %eigminlist(k)=min((diag(D)));
    eigmaxlist(k,l,m)=max((diag(D)));
end
end


figure(1)
for i = 1:length(tauc)
scatter(Hlist,eigmaxlist(:,i,2),30,"filled");
hold on;
end
hold off;
set(gca,"FontSize",20,"LineWidth",2.5,"Box","on");
legend("\bf \tau_{c} = 2","\bf \tau_{c} = 4","\bf \tau_{c} = 6","\bf \tau_{c} = 8","\bf \tau_{c} = 10","\bf \tau_{c} = 12","Location","southeast");
pbaspect([4 3 1]);
figure(2)

for k = 1:2^N
    for m= 1:length(tau)
plot(tauc,eigmaxlist(k,:,m),'LineWidth',2.5);
hold on
    end
end
hold off
set(gca,"FontSize",20,"LineWidth",2.5)
legend("\bf \tau = 0.5","\bf \tau = 1","\bf \tau = 1.5","Location","southeast");
pbaspect([4 3 1]);
%legend on;


%OIM



dec = [0:2^N-1]';
bin=( decimalToBinaryVector(dec));
theta=bin*pi;
S = 2*bin-ones(2^N,N);
HlistOIM=zeros(2^N,1);
for i=1:2^N
    HlistOIM(i)=-0.5*S(i,:)*J*S(i,:)';
end
JacobianOIM = zeros(N,N);

eigmaxlistOIM = zeros(2^N,1);
K=1;
Ks=1.5;
for k =1:2^N
    parfor i = 1:N
        for j = 1:N
             if i==j
                JacobianOIM(i,j)= Energy(K,J,Ks,theta,N,i,k);
            else
                JacobianOIM(i,j) = K*J(i,j)*cos(theta(k,i)-theta(k,j));
             end
        end
    end
    [V,D]=eig(JacobianOIM);
   % eigminlist(k)=min((diag(D)));
    eigmaxlistOIM(k)=max((diag(D)));
end

scatter(HlistOIM,eigmaxlistOIM,20,'filled');
set(gca,"FontSize",20,"LineWidth",2.5,"Box","on");

figure('Units','inches','Position',[1 1 3.5 2.4],'Color','w','Renderer','painters');

t= tiledlayout(1,2,"TileSpacing","compact","Padding","compact");
labels = {'(a)','(b)'};
colors = lines(length(tauc));
ax = nexttile(1);

for i = 1:length(tauc)
scatter(Hlist,eigmaxlist(:,i,2),2,"filled","MarkerEdgeColor",colors(i,:),"MarkerFaceColor",colors(i,:),"LineWidth",0.7);
hold on;
end
hold off;
%set(gca,"FontSize",20,"LineWidth",2.5,"Box","on");
ax.LineWidth = 1;
ax.FontSize = 10;
ax.FontWeight="normal"
ax.TickDir = 'in';
ax.Box = "on"

lgd=legend("\bf \tau_{c} = 4","\bf \tau_{c} = 12","\bf \tau_{c} = 24","Location","best","Orientation","vertical");
lgd.FontSize = 8;
%lgd.Interpreter = "latex";
lgd.Box = "off";
text(ax,0.03,0.95,labels(1),'Units','normalized','FontSize',9,'Color',[0.1 0.1 0.1]);
title("BLIM","FontWeight","bold");

xlabel("H","FontSize",14,"FontWeight","normal");
ylabel("\lambda_{max}","FontSize",12,"FontWeight","normal");
xlim([min(Hlist)-2 max(Hlist)+2] )
ylim([-1.42 -1] )
%pbaspect(ax,[16 9 1]);

ax =nexttile(2);

scatter(HlistOIM,eigmaxlistOIM,0.8,'filled',"MarkerEdgeColor",colors(1,:),"MarkerFaceColor",colors(1,:),"LineWidth",0.7);
%set(gca,"FontSize",20,"LineWidth",2.5,"Box","on");
text(ax,0.03,0.95,labels(2),'Units','normalized','FontSize',9,'Color',[0.1 0.1 0.1]);
title("OIM","FontWeight","bold");
%pbaspect(ax,[16 9 1]);
ax.LineWidth = 1;
ax.FontSize = 10;
ax.FontWeight="normal"
ax.TickDir = 'in';
ax.Box = "on"
xlim([min(HlistOIM)-5 max(HlistOIM)+5] )
xlabel(ax,"H","FontSize",14,"FontWeight","normal");
ylabel(ax,"\lambda_{max}","FontSize",12,"FontWeight","normal");

%ylim([min(eigmaxlist)-0.06 max(eigmaxlist)+0.06] )


%title(t,"Corresponding eigenvalues of different energy configurations","FontSize",7,"FontWeight","bold");

exportgraphics(gcf,"D:\Abir\BLIM\Datav3\EigvsHam.pdf","ContentType","vector","Resolution",600,"BackgroundColor","white");

function E = Energy(K,J,Ks,theta,N,index,k)
E=0;
dtheta = theta(k,index)*ones(N,1)-theta(k,:)';
E = -K *J(index,:)*cos(dtheta)-2*Ks*cos(2*theta(k,index));
end


