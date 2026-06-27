%% ========================================================================
%  Parameter sweep with unified y-axis limits:
%    OIM  -- sweep K_s in {0.1, 0.5, 1, 1.5, 2}
%    BLIM -- sweep tau in {0.1, 0.5, 1, 10}   (k = 6, tau_c = 12 fixed)
%  All OIM plots share one y-limit;  all BLIM plots share another.
%  Limits are padded so no point sits on the border.
%  ========================================================================
clc; clear; close all

%% ======================== Shared setup ==================================
nOsc = 15;

J = round(rand(nOsc));
J = tril(J,-1);
J = -J - J.';
J = load("D:\Abir\BLIM\Graphs\G201.txt")
W = -J;

A_idx = 0:2^nOsc-1;
B = dec2bin(A_idx, nOsc) - '0';
S = 1 - 2*B;
nCfg = size(B,1);

H = zeros(1, nCfg);
for i = 1:nCfg
    H(i) = -0.5 * S(i,:) * J * S(i,:)';
end

An    = 1e-1;
tstop = 10;
tstep = 5e-3;

outdir = fullfile('D:\Abir\BLIM\Datav3\Configuration initialization Review');
if ~exist(outdir, 'dir'); mkdir(outdir); end

BOX_LW    = 2;
FONT_SZ   = 28;
LABEL_SZ  = 32;
PAD_FRAC  = 0.05;   % fractional padding so points don't sit on the border

% -------- User-tunable axis limits and tick locations --------
% Set a value to [] to auto-compute from data (with PAD_FRAC padding).
% Set XTicks/YTicks to [] to let MATLAB choose automatically.

% OIM
OIM_XLim  = [];                % e.g. [-25 50],  [] = auto
OIM_YLim  = [-8,15];                % e.g. [-1 15],   [] = auto
OIM_XTick = [];                % e.g. [-20 0 20 40],  [] = auto
OIM_YTick = [-5 0 5 10 15];

% BLIM
BLIM_XLim  = [];
BLIM_YLim  = [-11,1];
BLIM_XTick = [];
BLIM_YTick = [];

%% ======================== OIM sweep (compute phase) =====================
Ks_list = [1,2.5,4];
K_oim   = 1;
Ac      = K_oim;

nKs = numel(Ks_list);
lambda_max_oim_all = zeros(nKs, nCfg);
H_delta_oim_all    = zeros(nKs, nCfg);

for s = 1:nKs
    Ks_oim = Ks_list(s);
    As     = Ks_oim;

    lambda_max_oim = zeros(1, nCfg);
    spin_final_oim = zeros(nCfg, nOsc);
    H_final_oim    = zeros(1, nCfg);

    parfor k = 1:nCfg
        Ja = zeros(nOsc);
        for i = 1:nOsc
            for j = 1:nOsc
                if i ~= j
                    Ja(i,j) = -K_oim * W(i,j) * cos(pi*(B(k,i)-B(k,j)));
                end
            end
            T = 0;
            for jj = 1:nOsc
                T = T + K_oim * W(i,jj) * cos(pi*(B(k,i)-B(k,jj)));
            end
            Ja(i,i) = T - 2 * Ks_oim * cos(2*pi*B(k,i));
        end
        ev = eig(Ja);
        [~, idx] = max(real(ev));
        lambda_max_oim(k) = real(ev(idx));

        initial = B(k,:)';
        F1 = @(t,X) KuramotoF1(X, Ac, As, nOsc, J);
        G1 = @(t,X) An * eye(nOsc);
        obj1 = sde(F1, G1, 'StartState', initial);
        [S1, ~] = simulate(obj1, tstop/tstep, 'DeltaTime', tstep);

        spin_final_oim(k,:) = sign(cos(pi*S1(end,:)));
        H_final_oim(k) = -0.5 * spin_final_oim(k,:) * J * spin_final_oim(k,:)';
    end

    lambda_max_oim_all(s,:) = lambda_max_oim;
    H_delta_oim_all(s,:)    = H_final_oim - H;

    fprintf('OIM:  Ks = %.2f computed\n', Ks_oim);
end

%% ======================== BLIM sweep (compute phase) ====================
tau_list = [0.1,0.5,1];   % sweep the self time constant tau
kgain    = 10;                   % fixed tanh gain k
tauc     = 30;                  % fixed coupling time constant

nTau = numel(tau_list);
lambda_max_blim_all = zeros(nTau, nCfg);
H_delta_blim_all    = zeros(nTau, nCfg);

for s = 1:nTau
    tau = tau_list(s);

    lambda_max_blim = zeros(1, nCfg);
    spin_final_blim = zeros(nCfg, nOsc);
    H_final_blim    = zeros(1, nCfg);

    parfor k = 1:nCfg
        s_k = S(k,:)';

        th_in     = tanh(kgain * s_k);
        sech2_in  = sech(kgain * s_k).^2;
        sech2_out = sech(kgain * th_in).^2;
        gprime    = (kgain^2) * sech2_out .* sech2_in;
        rowsumJ   = sum(J,2);

        Ja = zeros(nOsc);
        for i = 1:nOsc
            for j = 1:nOsc
                if i ~= j
                    Ja(i,j) = (J(i,j)/tauc) ;
                end
            end
            Ja(i,i) = -(1/tau)*(gprime(i) + 1) + (1/tauc)*rowsumJ(i);
        end
        ev = eig(Ja);
        [~, idx] = max(real(ev));
        lambda_max_blim(k) = real(ev(idx));

        initial = s_k;
        F1 = @(t,X) tanhNestedF1(X, tau, tauc, kgain, J);
        G1 = @(t,X) An * eye(nOsc);
        obj1 = sde(F1, G1, 'StartState', initial);
        [S1, ~] = simulate(obj1, tstop/tstep, 'DeltaTime', tstep);

        sf = sign(S1(end,:));
        sf(sf==0) = 1;
        spin_final_blim(k,:) = sf;
        H_final_blim(k) = -0.5 * sf * J * sf';
    end

    lambda_max_blim_all(s,:) = lambda_max_blim;
    H_delta_blim_all(s,:)    = H_final_blim - H;

    fprintf('BLIM: tau = %.2f computed\n', tau);
end

%% ======================== Global axis limits ============================
% Auto-compute from data; user overrides take precedence if provided.
xl_auto   = padLimits([min(H), max(H)], PAD_FRAC);
yl_oim_a  = padLimits([min(lambda_max_oim_all(:)),  max(lambda_max_oim_all(:))],  PAD_FRAC);
yl_blim_a = padLimits([min(lambda_max_blim_all(:)), max(lambda_max_blim_all(:))], PAD_FRAC);

xl_oim   = pickLimits(OIM_XLim,  xl_auto);
yl_oim   = pickLimits(OIM_YLim,  yl_oim_a);
xl_blim  = pickLimits(BLIM_XLim, xl_auto);
yl_blim  = pickLimits(BLIM_YLim, yl_blim_a);

fprintf('\nOIM  x-limits: [%.3f, %.3f]   y-limits: [%.3f, %.3f]\n', ...
        xl_oim(1), xl_oim(2), yl_oim(1), yl_oim(2));
fprintf('BLIM x-limits: [%.3f, %.3f]   y-limits: [%.3f, %.3f]\n', ...
        xl_blim(1), xl_blim(2), yl_blim(1), yl_blim(2));

%% ======================== OIM plotting phase ============================
for s = 1:nKs
    Ks_oim = Ks_list(s);
    C_oim  = makeColors(H_delta_oim_all(s,:));

    fig = figure('Color','w','Position',[100 100 600 560]);
    ax  = axes('Parent',fig,'Color','w','LineWidth',BOX_LW);
    hold(ax,'on')
    scatter(ax, H, lambda_max_oim_all(s,:), 50, C_oim, 'filled')
    axis(ax,'square')
    set(ax, 'FontSize', FONT_SZ, 'LineWidth', BOX_LW, ...
            'Box','off', 'TickDir','out', ...
            'XLim', xl_oim, 'YLim', yl_oim)
    applyTicks(ax, OIM_XTick, OIM_YTick);
    xlabel(ax,'H', 'FontSize', LABEL_SZ)
    ylabel(ax,'\lambda_{max}', 'Interpreter', 'tex', ...
           'FontSize', LABEL_SZ)
    drawBoxNoMirrorTicks(ax, BOX_LW)
    set(fig,'InvertHardcopy','off')

    fname = sprintf('OIM_Ks_%s', strrep(num2str(Ks_oim,'%.2f'), '.', 'p'));
    exportgraphics(fig, fullfile(outdir, [fname '.png']), 'Resolution', 600, ...
                   'BackgroundColor', 'white');
    exportgraphics(fig, fullfile(outdir, [fname '.pdf']), ...
                   'ContentType', 'vector', 'BackgroundColor', 'white');
    fprintf('Saved %s.{png,pdf}\n', fname);
end

%% ======================== BLIM plotting phase ===========================
for s = 1:nTau
    tau    = tau_list(s);
    C_blim = makeColors(H_delta_blim_all(s,:));

    fig = figure('Color','w','Position',[100 100 600 560]);
    ax  = axes('Parent',fig,'Color','w','LineWidth',BOX_LW);
    hold(ax,'on')
    scatter(ax, H, lambda_max_blim_all(s,:), 50, C_blim, 'filled')
    axis(ax,'square')
    set(ax, 'FontSize', FONT_SZ, 'LineWidth', BOX_LW, ...
            'Box','off', 'TickDir','out', ...
            'XLim', xl_blim, 'YLim', yl_blim)
    applyTicks(ax, BLIM_XTick, BLIM_YTick);
    xlabel(ax,'H', 'FontSize', LABEL_SZ)
    ylabel(ax,'\lambda_{max}', 'Interpreter', 'tex', ...
           'FontSize', LABEL_SZ)
    drawBoxNoMirrorTicks(ax, BOX_LW)
    set(fig,'InvertHardcopy','off')

    fname = sprintf('BLIM_tau_%s', strrep(num2str(tau,'%.2f'), '.', 'p'));
    exportgraphics(fig, fullfile(outdir, [fname '.png']), 'Resolution', 600, ...
                   'BackgroundColor', 'white');
    exportgraphics(fig, fullfile(outdir, [fname '.pdf']), ...
                   'ContentType', 'vector', 'BackgroundColor', 'white');
    fprintf('Saved %s.{png,pdf}\n', fname);
end

%% ======================== Trajectory figures ============================
% Trace state trajectories of ALL oscillators vs time for ONE
% representative initial configuration (the config at median H).
% Parameter values for these single runs:
Ks_traj  = 1;     % K_s used for the OIM trajectory run
tau_traj = 15;     % tau used for the BLIM trajectory run
                  % (k = kgain, tauc = 12 are fixed in the BLIM sweep)

% Pick the representative configuration: index of the median-H config.
[~, order] = sort(H);
rep_idx    = order(ceil(numel(order)/2));    % median-H configuration index

TRAJ_LW = 1.5;     % trajectory line width

% --- OIM trajectory ---
Ac = K_oim;  As = Ks_traj;
initial = B(rep_idx,:)';
F1 = @(t,X) KuramotoF1(X, Ac, As, nOsc, J);
G1 = @(t,X) An * eye(nOsc);
obj = sde(F1, G1, 'StartState', initial);
[Soim, Toim] = simulate(obj, tstop/tstep, 'DeltaTime', tstep);

fig = figure('Color','w','Position',[100 100 640 520]);
ax  = axes('Parent',fig,'Color','w','LineWidth',BOX_LW);
hold(ax,'on')
plot(ax, Toim, Soim, 'LineWidth', TRAJ_LW)
axis(ax,'square')
set(ax, 'FontSize', FONT_SZ, 'LineWidth', BOX_LW, 'Box','off', 'TickDir','out')
xlabel(ax,'$t$', 'Interpreter','latex', 'FontSize', LABEL_SZ)
ylabel(ax,'$\phi_i$', 'Interpreter','latex', 'FontSize', LABEL_SZ)
drawBoxNoMirrorTicks(ax, BOX_LW)
set(fig,'InvertHardcopy','off')
exportgraphics(fig, fullfile(outdir, 'OIM_trajectory.png'), 'Resolution', 600, 'BackgroundColor','white');
exportgraphics(fig, fullfile(outdir, 'OIM_trajectory.pdf'), 'ContentType','vector', 'BackgroundColor','white');
fprintf('Saved OIM_trajectory.{png,pdf}  (Ks=%.2f, config #%d)\n', Ks_traj, rep_idx);

% --- BLIM trajectory ---
initial = S(rep_idx,:)';
F1 = @(t,X) tanhNestedF1(X, tau_traj, tauc, kgain, J);
G1 = @(t,X) An * eye(nOsc);
obj = sde(F1, G1, 'StartState', initial);
[Sblim, Tblim] = simulate(obj, tstop/tstep, 'DeltaTime', tstep);

fig = figure('Color','w','Position',[760 100 640 520]);
ax  = axes('Parent',fig,'Color','w','LineWidth',BOX_LW);
hold(ax,'on')
plot(ax, Tblim, Sblim, 'LineWidth', TRAJ_LW)
axis(ax,'square')
set(ax, 'FontSize', FONT_SZ, 'LineWidth', BOX_LW, 'Box','off', 'TickDir','out')
xlabel(ax,'$t$', 'Interpreter','latex', 'FontSize', LABEL_SZ)
ylabel(ax,'$v_i$', 'Interpreter','latex', 'FontSize', LABEL_SZ)
drawBoxNoMirrorTicks(ax, BOX_LW)
set(fig,'InvertHardcopy','off')
exportgraphics(fig, fullfile(outdir, 'BLIM_trajectory.png'), 'Resolution', 600, 'BackgroundColor','white');
exportgraphics(fig, fullfile(outdir, 'BLIM_trajectory.pdf'), 'ContentType','vector', 'BackgroundColor','white');
fprintf('Saved BLIM_trajectory.{png,pdf}  (tau=%.2f, config #%d)\n', tau_traj, rep_idx);

%% ======================== Functions =====================================
function fout = KuramotoF1(x, Ac, As, n, J)
    fout = zeros(n,1);
    for c = 1:n
        fout(c,1) = -Ac * J(c,:) * sin(pi*(x(c) - x));
    end
    fout = (fout - As * sin(2*pi*x)) / pi;
end

function fout = tanhNestedF1(v, tau, tauc, kgain, J)
    th_in  = tanh(kgain * v);
    th_out = tanh(kgain * th_in);
    self_term = (th_out - v) / tau;
    rowsumJ   = sum(J,2);
    coup_term = (rowsumJ .* v + J * th_in) / tauc;
    fout = self_term + coup_term;
end

function C = makeColors(H_delta)
    C = zeros(length(H_delta), 3);
    C(H_delta == 0, :) = repmat([1 0 0], sum(H_delta==0), 1);
    C(H_delta ~= 0, :) = repmat([0 0 1], sum(H_delta~=0), 1);
end

function lim = padLimits(rng, fracPad)
%PADLIMITS  Expand a [min,max] range by a symmetric fractional padding.
    lo = rng(1);  hi = rng(2);
    span = hi - lo;
    if span <= 0
        span = max(abs(lo), 1) * 1e-3;
    end
    pad = fracPad * span;
    lim = [lo - pad, hi + pad];
end

function lim = pickLimits(userLim, autoLim)
%PICKLIMITS  Return userLim if non-empty and well-formed, else autoLim.
    if isempty(userLim)
        lim = autoLim;
    else
        lim = userLim;
    end
end

function applyTicks(ax, xtk, ytk)
%APPLYTICKS  Set XTick / YTick on ax only when the user supplied a non-empty
%   vector.  Empty input leaves the auto-chosen ticks in place.
    if ~isempty(xtk)
        set(ax, 'XTick', xtk);
    end
    if ~isempty(ytk)
        set(ax, 'YTick', ytk);
    end
end

function drawBoxNoMirrorTicks(ax, lw)
    drawnow
    xl = ax.XLim;  yl = ax.YLim;
    line(ax, [xl(1) xl(2)], [yl(1) yl(1)], 'Color','k', 'LineWidth', lw, 'Clipping','off');
    line(ax, [xl(1) xl(2)], [yl(2) yl(2)], 'Color','k', 'LineWidth', lw, 'Clipping','off');
    line(ax, [xl(1) xl(1)], [yl(1) yl(2)], 'Color','k', 'LineWidth', lw, 'Clipping','off');
    line(ax, [xl(2) xl(2)], [yl(1) yl(2)], 'Color','k', 'LineWidth', lw, 'Clipping','off');
end