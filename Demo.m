% test for Rethinking Tensor Rotation in Multi-View Subspace Clustering 
% via Non-Rotated Compact Tensor Modeling Algorithm

clear
close all
addpath('./ClusteringMeasure', './Funs');

data_path = './Data/';

num_runs = 1; %number of runs

% Algorithm Settings

Data_name = 'MSRCv1';
views = 5;

%% Loading data
fprintf('Testing %s...\n',Data_name) 
load(fullfile(data_path, strcat(Data_name,'.mat')));
load(fullfile(data_path, strcat(Data_name,'_index.mat')));
gt0 = gt(:); 

for k=1:views
    eval(sprintf('X{%d} = double(X%d);', k, k));
end
X0=X;

cls_num = length(unique(gt));
K = length(X);

%% Records
alg_name = 'NRCT-MVSC'; 
alg_time = zeros(num_runs,1);
NMI = zeros(num_runs,1);
ACC = zeros(num_runs,1);
AR = zeros(num_runs,1);
fscore = zeros(num_runs,1);   
precision = zeros(num_runs,1);
recall = zeros(num_runs,1);
purity =zeros(num_runs,1);

C1 = cell(num_runs,1);     % clustering results
S1 = cell(num_runs,1);     % affinity matrices
Out1 = cell(num_runs,1);   % metrics

%% Algs Running

% Parameter settings
opts.alpha = 1e0;   %[1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1, 5, 10, 50, 100];
opts.beta = 1e0;    %[1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3];

opts.flag_debug = 0;
opts.maxIter = 200;
opts.epsilon = 1e-6;
opts.mu = 1e-5; 
opts.max_mu = 1e10; 
opts.pho_mu = 2;
opts.r = 30;        %[10, 20, ..., 100];

for kk = 1:num_runs
    index = index_all(kk, :);
    for k = 1:views
        X{k} = X0{k}(:, index);
    end
    gt = gt0(index);
    gt = gt(:);
    Y = X;
    for iv=1:K
        [Y{iv}]=NormalizeData(X{iv});
    end   
    time_start = tic;

    [C, S, Out] = solving_NRCT_MVSC(Y, cls_num, gt, opts);

    Out.time= toc(time_start);        
    alg_time(kk) =  Out.time;
    NMI(kk) = Out.NMI;
    AR(kk) = Out.AR;
    ACC(kk) = Out.ACC;
    recall(kk) = Out.recall;
    precision(kk) = Out.precision;
    fscore(kk) = Out.fscore; 
    purity(kk) = Out.purity; 
    C1{kk} = C;
    S1{kk} = S;
    Out1{kk} = Out;
end

%% Results report
fprintf('alpha=%.3f,beta=%.3f,r=%d\n',opts.alpha,opts.beta,opts.r);
fprintf('%6s\t%12s\t%4s\t%4s\t%4s\t%4s\t%4s\t%4s\t%4s\t%4s\n','Stats', ...\
 'Algs', 'Time', 'NMI', 'AR', 'ACC', 'Recall', 'Pre', 'F-Score', 'Purity');
fprintf('%6s\t%12s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n',...\
'Mean', alg_name,mean(alg_time),mean(NMI),mean(AR),...\
mean(ACC),mean(recall),mean(precision),mean(fscore),mean(purity));   
fprintf('%6s\t%12s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n',...\
'Std', alg_name,std(alg_time),std(NMI),std(AR),...\
std(ACC),std(recall),std(precision),std(fscore),std(purity));