clear; clc; close all
FS = 24;

%% vs epsilon
epsilon_all = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1, 5, 10];

load synth_vs_eps_D50_adam_v2
avg_nonp_tr = (acc_nonp_tr);
avg_nonp_ts = (acc_nonp_ts);
avg_conv_tr = (acc_conv_tr);
avg_conv_ts = (acc_conv_ts);
avg_cape_tr = (acc_cape_tr);
avg_cape_ts = (acc_cape_ts);

subplot(141)
semilogx(epsilon_all, avg_nonp_tr, 'r^-.','LineWidth',3,'MarkerSize',15); hold on
semilogx(epsilon_all, avg_cape_tr, 'kh--','LineWidth',3,'MarkerSize',15); hold on
semilogx(epsilon_all, avg_conv_tr, 'bx--','LineWidth',3,'MarkerSize',15); hold on

set(gca,'FontSize',FS,'FontWeight','bold')
axis([epsilon_all(1)/10 epsilon_all(end)*10 20 105])
xlabel('(a) Privacy parameter (\epsilon)','FontSize',FS,'FontWeight','bold');
ylabel('acc (%)','FontSize',FS,'FontWeight','bold');
title('Train set (D = 50, N = 10k)','FontSize', FS,'FontWeight','bold'); 
legend('Non-priv', 'capeFM', 'conv', 'Location','NE')

subplot(142)
semilogx(epsilon_all, avg_nonp_ts, 'r^-.','LineWidth',3,'MarkerSize',15); hold on
semilogx(epsilon_all, avg_cape_ts, 'kh--','LineWidth',3,'MarkerSize',15); hold on
semilogx(epsilon_all, avg_conv_ts, 'bx--','LineWidth',3,'MarkerSize',15); hold on

set(gca,'FontSize',FS,'FontWeight','bold')
axis([epsilon_all(1)/10 epsilon_all(end)*10 20 105])
xlabel('(b) Privacy parameter (\epsilon)','FontSize',FS,'FontWeight','bold');
ylabel('acc (%)','FontSize',FS,'FontWeight','bold');
title('Test set (D = 50, N = 10k)','FontSize', FS,'FontWeight','bold'); 


%% vs samples
N_all = [1000, 2000, 3000, 4000, 5000, 7000, 8000, 10000];
load synth_vs_samples_D50_adam_v2
avg_nonp_tr = (acc_nonp_tr);
avg_nonp_ts = (acc_nonp_ts);
avg_conv_tr = (acc_conv_tr);
avg_conv_ts = (acc_conv_ts);
avg_cape_tr = (acc_cape_tr);
avg_cape_ts = (acc_cape_ts);

subplot(143)
semilogx(N_all, avg_nonp_tr, 'r^-.','LineWidth',3,'MarkerSize',15); hold on
semilogx(N_all, avg_cape_tr, 'kh--','LineWidth',3,'MarkerSize',15); hold on
semilogx(N_all, avg_conv_tr, 'bx--','LineWidth',3,'MarkerSize',15); hold on

set(gca,'FontSize',FS,'FontWeight','bold')
axis([N_all(1)/2 N_all(end)*2 20 105])
xlabel('(c) Total samples (N)','FontSize',FS,'FontWeight','bold');
ylabel('acc (%)','FontSize',FS,'FontWeight','bold');
title('Train set (D = 50, \epsilon = 0.01)','FontSize', FS,'FontWeight','bold'); 

subplot(144)
semilogx(N_all, avg_nonp_ts, 'r^-.','LineWidth',3,'MarkerSize',15); hold on
semilogx(N_all, avg_cape_ts, 'kh--','LineWidth',3,'MarkerSize',15); hold on
semilogx(N_all, avg_conv_ts, 'bx--','LineWidth',3,'MarkerSize',15); hold on

set(gca,'FontSize',FS,'FontWeight','bold')
axis([N_all(1)/2 N_all(end)*2 20 105])
xlabel('(d) Total samples (N)','FontSize',FS,'FontWeight','bold');
ylabel('acc (%)','FontSize',FS,'FontWeight','bold');
title('Test set (D = 50, \epsilon = 0.01)','FontSize', FS,'FontWeight','bold'); 



