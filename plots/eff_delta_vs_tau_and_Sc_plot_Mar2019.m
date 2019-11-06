
clear; clc; close all
FS = 30;
delta_all = logspace(-5, -2, 15); delta_all = delta_all(:);
tau_all = logspace(-2, 2, 50); tau_all = tau_all(:);

%% new version - compare effective delta with tau
% S = 10, eps = 0.001
N = 1000;
epsilon_all = [1e-3];
S_all = [10]';
res = [];
res_fic_delta = [];

for s_id = 1:length(S_all)
    S = S_all(s_id);
    Ns = N / S;
    Sc = ceil(S/3) - 1;
    Sh = S - Sc;
    eff_delta = zeros(length(epsilon_all), length(tau_all));
    fic_delta = zeros(length(epsilon_all), length(tau_all));
    
    for eps_id = 1:length(epsilon_all)
        epsilon = epsilon_all(eps_id);
        for del_id = 1:length(tau_all)
%             delta = delta_all(del_id);
% 
%             Ns = N / S;
% 
%             tau = (1 / (Ns * epsilon)) * sqrt(2 * log(1.25/delta));
            tau = tau_all(del_id);
            Sigma_a = (1 + 1/S) * tau^2 * eye(Sh) - ones(Sh, Sh) * (tau^2 / S);
            Sigma_ae = ones(Sh, 1) * (1-Sh/S) * tau^2;
            Sigma_e = Sh * tau^2;
            Sigma = [Sigma_a, Sigma_ae; Sigma_ae', Sigma_e];
            v = [1/Ns; zeros(Sh, 1)];

            mu_z = 0.5 * v' * pinv(Sigma) * v;
            sigma_sq_z = v' * pinv(Sigma) * v;
            
            tmp = (epsilon - mu_z) / sqrt(sigma_sq_z);
            eff_delta(eps_id, del_id) = 2 * qfunc(tmp); 
            fic_delta(eps_id, del_id) = 1.25 / exp((S * (Ns * epsilon * tau / S) ^ 2) / 2);
        end
    end
    res = [res; eff_delta];
    res_fic_delta = [res_fic_delta; fic_delta];
end
subplot(121)
% tau_all = (1 / (Ns * epsilon)) * sqrt(2 * log(1.25./delta_all));
semilogy(tau_all, (res)', 'kh--', 'LineWidth', 3,'MarkerSize',15); hold on
semilogy(tau_all, (res_fic_delta)', 'bx--', 'LineWidth', 3,'MarkerSize',15)
set(gca,'FontSize',FS,'FontWeight','bold')
legend('\delta', '\delta_{conv}', 'location', 'best')
xlabel('Local noise \tau_s', 'FontSize', FS,'FontWeight','bold')
ylabel('\delta and \delta_{conv}', 'FontSize', FS,'FontWeight','bold')
title(['N = ', num2str(N), ', S = ', num2str(S), ', \epsilon = ', num2str(epsilon)], 'FontSize', FS,'FontWeight','bold')
axis([-0.1, 102, 1e-20, 2])


% S = 30, eps = 0.1
N = 5000;
epsilon_all = [1e-1];
S_all = [30]';
res = [];
res_fic_delta = [];

for s_id = 1:length(S_all)
    S = S_all(s_id);
    Ns = N / S;
    Sc = ceil(S/3) - 1;
    Sh = S - Sc;
    eff_delta = zeros(length(epsilon_all), length(tau_all));
    fic_delta = zeros(length(epsilon_all), length(tau_all));
    
    for eps_id = 1:length(epsilon_all)
        epsilon = epsilon_all(eps_id);
        for del_id = 1:length(tau_all)
%             delta = delta_all(del_id);
% 
%             
% 
%             tau = (1 / (Ns * epsilon)) * sqrt(2 * log(1.25/delta));
            tau = tau_all(del_id);
            Sigma_a = (1 + 1/S) * tau^2 * eye(Sh) - ones(Sh, Sh) * (tau^2 / S);
            Sigma_ae = ones(Sh, 1) * (1-Sh/S) * tau^2;
            Sigma_e = Sh * tau^2;
            Sigma = [Sigma_a, Sigma_ae; Sigma_ae', Sigma_e];
            v = [1/Ns; zeros(Sh, 1)];

            mu_z = 0.5 * v' * pinv(Sigma) * v;
            sigma_sq_z = v' * pinv(Sigma) * v;
            
            tmp = (epsilon - mu_z) / sqrt(sigma_sq_z);
            eff_delta(eps_id, del_id) = 2 * qfunc(tmp);
            fic_delta(eps_id, del_id) = 1.25 / exp((S * (Ns * epsilon * tau / S) ^ 2) / 2);
        end
    end
    res = [res; eff_delta];
    res_fic_delta = [res_fic_delta; fic_delta];
end
subplot(122)
% tau_all = (1 / (Ns * epsilon)) * sqrt(2 * log(1.25./delta_all));
semilogy(tau_all, (res)', 'kh--', 'LineWidth', 3,'MarkerSize',15); hold on
semilogy(tau_all, (res_fic_delta)', 'bx--', 'LineWidth', 3,'MarkerSize',15)
set(gca,'FontSize',FS,'FontWeight','bold')
% legend('\delta', '\delta_{conv}', 'location', 'best')
xlabel('Local noise \tau_s', 'FontSize', FS,'FontWeight','bold')
ylabel('\delta and \delta_{conv}', 'FontSize', FS,'FontWeight','bold')
title(['N = ', num2str(N), ', S = ', num2str(S), ', \epsilon = ', num2str(epsilon)], 'FontSize', FS,'FontWeight','bold')
axis([-0.1, 10, 1e-500, 2])

% subplot(122)
% loglog(tau_all, (res_fic_delta)', 'LineWidth', 3)
% set(gca,'FontSize',FS,'FontWeight','bold')
% legend(legend_str)
% xlabel('(a) $$\delta$$', 'Interpreter','latex', 'FontSize', FS,'FontWeight','bold')
% ylabel('$$\frac{\tilde{\delta}}{\delta_{conv}}$$', 'Interpreter','latex', 'FontSize', FS,'FontWeight','bold')
% title('$$S_C = \left\lceil \frac{S}{3} \right\rceil - 1$$', 'Interpreter','latex', 'FontSize', FS,'FontWeight','bold')
% % axis([delta_all(1) / 2, delta_all(end) * 2, 1e-6, 1e-2])

%% new version - compare effective delta with delta that we would have in conv setting
figure
% variation with Sc
N = 1000;
epsilon_all = [1e-3];
S_all = [10]';
tau = 50.0;

res = [];
res_fic_delta = [];

legend_str = cell(1, length(S_all) * length(epsilon_all));
for s_id = 1:length(S_all)
    S = S_all(s_id);
    Ns = N / S;
    
    Sc_all = [ceil(S/8), ceil(S/7), ceil(S/6), ceil(S/5), ceil(S/4), ceil(S/3) - 1]';
    eff_delta = zeros(length(epsilon_all), length(Sc_all));
    fic_delta = zeros(length(epsilon_all), length(Sc_all));
    
    for eps_id = 1:length(epsilon_all)
        epsilon = epsilon_all(eps_id);
        legend_str{1, (s_id - 1) * length(epsilon_all) + ...
            eps_id} = ['S = ', num2str(S), ', \epsilon = ', num2str(epsilon)];
        
        for sc_id = 1:length(Sc_all)
            Sc = Sc_all(sc_id);
            Sh = S - Sc;

            Sigma_a = (1 + 1/S) * tau^2 * eye(Sh) - ones(Sh, Sh) * (tau^2 / S);
            Sigma_ae = ones(Sh, 1) * (1-Sh/S) * tau^2;
            Sigma_e = Sh * tau^2;
            Sigma = [Sigma_a, Sigma_ae; Sigma_ae', Sigma_e];
            v = [1/Ns; zeros(Sh, 1)];

            mu_z = 0.5 * v' * pinv(Sigma) * v;
            sigma_sq_z = v' * pinv(Sigma) * v;
            
            tmp = (epsilon - mu_z) / sqrt(sigma_sq_z);
            eff_delta(eps_id, sc_id) = 2 * qfunc(tmp);
            fic_delta(eps_id, sc_id) = 1.25 / exp((S * (Ns * epsilon * tau / S) ^ 2) / 2);
        end
    end
    res = [res; eff_delta];
    res_fic_delta = [res_fic_delta; fic_delta];
end
subplot(121)
fracs = [1/8, 1/7, 1/6, 1/5, 1/4, 1/3]';
semilogy(fracs, (res)', 'kh--', 'LineWidth', 3,'MarkerSize',15); hold on
semilogy(fracs, (res_fic_delta)', 'bx--', 'LineWidth', 3,'MarkerSize',15)
set(gca,'FontSize',FS,'FontWeight','bold')
legend('\delta', '\delta_{conv}', 'location', 'best')
xlabel('S_C / S', 'FontSize', FS,'FontWeight','bold')
ylabel('\delta and \delta_{conv}', 'FontSize', FS,'FontWeight','bold')
title(['N = ', num2str(N), ', S = ', num2str(S), ', \epsilon = ', num2str(epsilon), ', \tau_s = ', num2str(tau)], ...
    'FontSize', FS,'FontWeight','bold')
% axis([0.1, 0.35, 1e-4, 1e-2])

% variation with Sc
N = 5000;
epsilon_all = [1e-1];
S_all = [30]';
tau = 0.5;

res = [];
res_fic_delta = [];

legend_str = cell(1, length(S_all) * length(epsilon_all));
for s_id = 1:length(S_all)
    S = S_all(s_id);
    Ns = N / S;
    
    Sc_all = [ceil(S/8), ceil(S/7), ceil(S/6), ceil(S/5), ceil(S/4), ceil(S/3) - 1]';
    eff_delta = zeros(length(epsilon_all), length(Sc_all));
    fic_delta = zeros(length(epsilon_all), length(Sc_all));
    
    for eps_id = 1:length(epsilon_all)
        epsilon = epsilon_all(eps_id);
        legend_str{1, (s_id - 1) * length(epsilon_all) + ...
            eps_id} = ['S = ', num2str(S), ', \epsilon = ', num2str(epsilon)];
        
        for sc_id = 1:length(Sc_all)
            Sc = Sc_all(sc_id);
            Sh = S - Sc;

            Sigma_a = (1 + 1/S) * tau^2 * eye(Sh) - ones(Sh, Sh) * (tau^2 / S);
            Sigma_ae = ones(Sh, 1) * (1-Sh/S) * tau^2;
            Sigma_e = Sh * tau^2;
            Sigma = [Sigma_a, Sigma_ae; Sigma_ae', Sigma_e];
            v = [1/Ns; zeros(Sh, 1)];

            mu_z = 0.5 * v' * pinv(Sigma) * v;
            sigma_sq_z = v' * pinv(Sigma) * v;
            
            tmp = (epsilon - mu_z) / sqrt(sigma_sq_z);
            eff_delta(eps_id, sc_id) = 2 * qfunc(tmp);
            fic_delta(eps_id, sc_id) = 1.25 / exp((S * (Ns * epsilon * tau / S) ^ 2) / 2);
        end
    end
    res = [res; eff_delta];
    res_fic_delta = [res_fic_delta; fic_delta];
end
subplot(122)
fracs = [1/8, 1/7, 1/6, 1/5, 1/4, 1/3]';
semilogy(fracs, (res)', 'kh--', 'LineWidth', 3,'MarkerSize',15); hold on
semilogy(fracs, (res_fic_delta)', 'bx--', 'LineWidth', 3,'MarkerSize',15)
set(gca,'FontSize',FS,'FontWeight','bold')
% legend('\delta', '\delta_{conv}', 'location', 'best')
xlabel('S_C / S', 'FontSize', FS,'FontWeight','bold')
ylabel('\delta and \delta_{conv}', 'FontSize', FS,'FontWeight','bold')
title(['N = ', num2str(N), ', S = ', num2str(S), ', \epsilon = ', num2str(epsilon), ', \tau_s = ', num2str(tau)], ...
    'FontSize', FS,'FontWeight','bold')
% axis([0.1, 0.35, 1e-4, 1e-2])
