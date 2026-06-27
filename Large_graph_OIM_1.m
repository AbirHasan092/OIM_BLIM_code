function out = Large_graph_1(graphNum, jobid, outdir)


if nargin < 1 || isempty(graphNum)
    graphNum = 1;
end

if nargin < 2 || isempty(jobid)
    jobid = char(string(datetime("now", "Format", "yyyyMMdd_HHmmss")));
end

if nargin < 3 || isempty(outdir)
    outdir = pwd;
end

jobid = char(string(jobid));
outdir = char(string(outdir));

if ~isfolder(outdir)
    mkdir(outdir);
end

oldFolder = pwd;

setenv('OIM_GRAPH_NUM', num2str(graphNum));
setenv('OIM_JOBID', jobid);
setenv('OIM_OUTDIR', outdir);
setenv('OIM_OLD_FOLDER', oldFolder);

cd(outdir);

diary(fullfile(outdir, ['OIM_G' num2str(graphNum) '_' jobid '.out']));
diary on

clc
% clear
close all
% parpool(32)

clear all
rand_samples=0;

%=======Graph=====================
graphs=str2double(getenv('OIM_GRAPH_NUM'));

d = readmatrix(['G' num2str(graphs) '.mtx'], 'FileType', 'text'); 
    p = d(2:end,1);
    n = d(2:end,2);
    w = d(2:end,3);

    nOsc = max(max(p), max(n));
    h = zeros(nOsc, 1);
    W = sparse(p, n, w, nOsc, nOsc);
    J = -W - W';


%d=readmatrix(['G' num2str(graphs) '.mtx']);

% d=readmatrix('G22.txt');
%p=d(:,1);
%n=d(:,2);
%w=d(:,3);

%nOsc=max(n);
%h = zeros(nOsc, 1);
%W = sparse(p, n, w, nOsc, nOsc);
%J = - W - W.';

W=-J;


%=======Random_samples=====================
if rand_samples==1
    n_samples1=100;
    B1=round(rand(n_samples1,nOsc));
    %==============H=====================================
    for i=1:size(B1,1)
        H1(i)=-0.5*cos(pi*B1(i,:))*J*cos(pi*B1(i,:))';
    end
else
    B1=[];
end

%-----------------------------------------
B2=[];
n_restart=1;
n_samples2=[20000 100 200 200 500 1000 100 100 200 200 500 1000 500 1000 500 1000 500 1000 500 1000];
skip=3;

for j=1:n_restart

    % n_samples2=100;
    spin=sign(2*rand(1,nOsc)-1);
    beta=1;
    B2_t=spin;

    for i=1:n_samples2(j)*skip
        r=randi(nOsc);
        spin(r)=sign(tanh(beta*J(r,:)*spin')-(2*rand-1));

        if mod(i,skip)==0
            B2_t=[B2_t;spin];
        end

    end

    B2=[B2; B2_t];

end

B2=0.5*(1+B2);

%==============H=====================================
for i=1:size(B2,1)
    H2(i)=-0.5*cos(pi*B2(i,:))*J*cos(pi*B2(i,:))';
end


%-------------------------------------------------
B=[B1;B2];

%==============H=====================================
for i=1:size(B,1)
    H(i)=-0.5*cos(pi*B(i,:))*J*cos(pi*B(i,:))';
end

K=1; %1
Ks=10;%10

%==============Jacobian=====================================
parfor k=1:size(B,1)
    
    %--------Calculate Jacobian--------------
    
    Ja = zeros(nOsc);
    e=zeros(size(B,1),nOsc);

    for i = 1:nOsc
        for j = 1:nOsc
            if i ~= j
                Ja(i,j) = K * J(i,j) * cos(pi*(B(k,i)-B(k,j))); %W
            end
        end

        % diagonal term
        T = 0;
        for jj = 1:nOsc
            T = T + K * W(i,jj) * cos(pi*(B(k,i)-B(k,jj)));
        end

        Ja(i,i) = T - 2 * Ks * cos(2*pi*B(k,i));
    end

    
    %===========Calculate Lambda_max===================
    
    e(k,:)=eig(Ja);
    lambda_max(k)=max(real(e(k,end)));

    %===========Now initialize the simulation===================

    Ac = K;
    An = 1e-6;
    tstop = 30; 
    tstep = 5e-3;
    As=Ks;

    initial=B(k,:)';

    F1 = @(t,X) KuramotoF1(X, Ac, As, nOsc,J);
    G1 = @(t,X) An*eye(nOsc);
    obj1 = sde(F1, G1, 'StartState', initial);
    [S1, T1] = simulate(obj1, tstop/tstep, 'DeltaTime', tstep);

    %-----------------

    % h = figure;
    % plot(T1, S1)
    % 
    % drawnow
    % pause(1)
    % 
    % close(h)

    %-----------------------

    spin_final(k,:)=sign(cos(pi*S1(length(T1),:)));

    H_final(k)=-0.5*spin_final(k,:)*J*spin_final(k,:)';

    H_delta(k)=H_final(k)-H(k);

end

H_compiled=[H' H_final'];


C = zeros(length(H_delta),3);   % initialize color matrix

C(H_delta == 0, :) = repmat([1 0 0], sum(H_delta==0), 1);   % red
C(H_delta ~= 0, :) = repmat([0 0 1], sum(H_delta~=0), 1);   % blue


%% ======================== Shared figure style ==========================
plotFontSize = 24;
plotLineWidth = 0.5;
plotTickDir = 'in';
plotGrid = 'off';

% Extra graph-specific saves for Slurm/multinode batch jobs.
graphNum = str2double(getenv('OIM_GRAPH_NUM'));
jobid = getenv('OIM_JOBID');
outdir = getenv('OIM_OUTDIR');
oldFolder = getenv('OIM_OLD_FOLDER');

%% ======================== Lambda vs H figure =========================
figLambda = figure;

scatter(H, lambda_max, 50, C, 'filled')

axLambda = gca;
box(axLambda, 'on')
axis(axLambda, 'square')
grid(axLambda, plotGrid)

set(axLambda, ...
    'FontSize', plotFontSize, ...
    'LineWidth', plotLineWidth, ...
    'TickDir', plotTickDir)

xlabel(axLambda, 'H')
ylabel(axLambda, '$\lambda_{\mathrm{max}}$', 'Interpreter', 'latex')

save("G1_large_graph")

save(fullfile(outdir, ['OIM_G' num2str(graphNum) '.mat']))
savefig(figLambda, fullfile(outdir, ['OIM_G' num2str(graphNum) '.fig']))
exportgraphics(figLambda, fullfile(outdir, ['OIM_G' num2str(graphNum) '.png']), 'Resolution', 300)

fprintf('Saved Slurm output file: %s\n', fullfile(outdir, ['OIM_G' num2str(graphNum) '.mat']));
fprintf('Saved Slurm figure file: %s\n', fullfile(outdir, ['OIM_G' num2str(graphNum) '.fig']));
fprintf('Saved Slurm PNG file: %s\n', fullfile(outdir, ['OIM_G' num2str(graphNum) '.png']));

%% ======================== OIM indicator plotting phase ==================
% y = 1[H_delta ~= 0] : 0 when the energy is unchanged, 1 otherwise.

ind = double(H_delta ~= 0);

figInd = figure;

scatter(H, ind, 50, C, 'filled')

axInd = gca;
box(axInd, 'on')
axis(axInd, 'square')
grid(axInd, plotGrid)

set(axInd, ...
    'FontSize', plotFontSize, ...
    'LineWidth', plotLineWidth, ...
    'TickDir', plotTickDir, ...
    'YLim', [-0.15 1.15], ...
    'YTick', [0 1])

xlabel(axInd, 'H')
ylabel(axInd, '$I(\Delta H \neq 0)$', 'Interpreter', 'latex')

indicatorBase = ['OIM_G' num2str(graphNum) '_indicator'];

savefig(figInd, fullfile(outdir, [indicatorBase '.fig']));

exportgraphics(figInd, fullfile(outdir, [indicatorBase '.png']), ...
               'Resolution', 300);

exportgraphics(figInd, fullfile(outdir, [indicatorBase '.pdf']), ...
               'ContentType', 'vector');

fprintf('Saved indicator FIG file: %s\n', fullfile(outdir, [indicatorBase '.fig']));
fprintf('Saved indicator PNG file: %s\n', fullfile(outdir, [indicatorBase '.png']));
fprintf('Saved indicator PDF file: %s\n', fullfile(outdir, [indicatorBase '.pdf']));

diary off

if ~isempty(oldFolder) && isfolder(oldFolder)
    cd(oldFolder);
end

out = "Done";

end









function fout = KuramotoF1(x, Ac, As, n, J)
for c = 1:n
    fout(c, 1) = -Ac * J(c, :) * sin(pi*(x(c) - x));
end
fout = (fout - As * sin(2*pi*x))/pi;
end