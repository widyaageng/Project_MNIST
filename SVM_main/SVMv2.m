clear;clc;

%%
%------DATA STRUCTURE--------%

% training image data structure
im_train = (fread(fopen('train-images.idx3-ubyte','r'),inf,'uint8'));
im_train_header = cast(im_train(1:16),'uint8');
im_train_header = reshape(dec2hex(im_train_header).',[8 4])';
im_train_count = hex2dec(im_train_header(2,:));
im_train = reshape(im_train(17:size(im_train)),[28*28 im_train_count]);
clear im_train_header;

% training label data structure
lab_train = (fread(fopen('train-labels.idx1-ubyte','r'),inf,'uint8'));
lab_train = lab_train(9:size(lab_train));
clear lab_train_header;

% test image data structure
im_test = (fread(fopen('t10k-images.idx3-ubyte','r'),inf,'uint8'));
im_test_header = cast(im_test(1:16),'uint8');
im_test_header = reshape(dec2hex(im_test_header).',[8 4])';
im_test_count = hex2dec(im_test_header(2,:));
im_test = reshape(im_test(17:size(im_test)), [28*28 im_test_count]);
clear im_test_header

% test label data structure
lab_test = (fread(fopen('t10k-labels.idx1-ubyte','r'),inf,'uint8'));
lab_test = lab_test(9:size(lab_test));
clear lab_test_header;

% trimming traindata and testdata for coding purpose
im_train = im_train(:,1:6000);
lab_train = lab_train(1:6000);
im_test = im_test(:,1:1000);
lab_test = lab_test(1:1000);
im_train_count = 6000;
im_test_count = 1000;



%---------- Calculating Covariance Matrix on Training Data ---------
% Calculate mean image, its size will be 784 by 1
im_train_mean = mean(im_train,2);
im_train_cov = zeros(size(im_train,1),size(im_train,1));
h = waitbar(0,'Please wait...');
index = 0;
for i=1:im_train_count
    index = index + 1;
    im_train_cov = im_train_cov + (im_train(:,i)-im_train_mean)*(im_train(:,i)-im_train_mean)';
    waitbar(index/(length(im_train)),h,sprintf('Generating cov matrix...%2.1f%%',100*index/(length(im_train))));
end
close(h);
% load('im_train_cov.mat');

%----------- Calculating eigenvalues and eigenvector matrix U -----------
[train_U,train_eig] = eig(im_train_cov);
train_eig = diag(train_eig);

% sorting eigenvalue and eigenvectors
[train_eig, idx_eig] = sort(train_eig,'descend');
train_dump = train_U(:,idx_eig(1));
for i = 2:length(idx_eig)
    train_dump = [train_dump train_U(:,idx_eig(i))];
end
train_U = train_dump;
clear train_dump;

%% GENERIC SVM SETTING
% margin setting
soft_margin_c = [0.01 0.1 1 10];

% radial basis kernel spread/variance
rbf_sigma = 1850;

% capturing accuracy for linear kernel,for different soft margin
acc40_lin = zeros(1,length(soft_margin_c));
acc80_lin = zeros(1,length(soft_margin_c));
acc200_lin = zeros(1,length(soft_margin_c));

% capturing accuracy for radial kernel, row=softmargin, col=radial spread
acc40_rbf = zeros(1,length(soft_margin_c));
acc80_rbf = zeros(1,length(soft_margin_c));
acc200_rbf = zeros(1,length(soft_margin_c));

%% PCA dimension reduction to 40 PCs and apply SVM
%-------------- 40 PCs Projection --------------
pc40 = train_U(:,1:40);

%-------------- Projecting Data ---------------
xpc40 = {}; % projected train image
ypc40 = {}; % projected test image
for i = 1:size(pc40,2)
    xpc40{i} = pc40(:,i)'*im_train;
    ypc40{i} = pc40(:,i)'*im_test;
end
xpc40 = cell2mat(xpc40');
ypc40 = cell2mat(ypc40');

% dummy train and test label for multiple binary classification, PC40
lab_train40_exc = lab_train.*zeros(size(xpc40,2),1);
lab_test40_exc = lab_test.*zeros(size(ypc40,2),1);

%% PCA dimension reduction to 80
%-------------- 80 PCs Projection --------------
pc80 = train_U(:,1:80);

%-------------- Projecting Data ---------------
xpc80 = {}; % projected train image
ypc80 = {}; % projected test image
for i = 1:size(pc80,2)
    xpc80{i} = pc80(:,i)'*im_train;
    ypc80{i} = pc80(:,i)'*im_test;
end
xpc80 = cell2mat(xpc80');
ypc80 = cell2mat(ypc80');

% dummy train and test label for multiple binary classification, PC80
lab_train80_exc = lab_train*zeros(1,10);
lab_test80_exc = lab_test*zeros(1,10);

%% PCA dimension reduction to 200
%-------------- 200 PCs Projection --------------
pc200 = train_U(:,1:200);

%-------------- Projecting Data ---------------
xpc200 = {}; % projected train image
ypc200 = {}; % projected test image
for i = 1:size(pc200,2)
    xpc200{i} = pc200(:,i)'*im_train;
    ypc200{i} = pc200(:,i)'*im_test;
end
xpc200 = cell2mat(xpc200');
ypc200 = cell2mat(ypc200');

% SVM for projected data to 200 PCA dimension
% Finding soft hyperplane for the 10 class data
lab_train200_exc = lab_train*zeros(1,10);
lab_test200_exc = lab_test*zeros(1,10);

%%

% initiating weight & bias for each class binary and each PCA data compression as
% well as kernel

weight40_class_rbf = zeros(size(pc40,2),10);
weight40_class_lin = weight40_class_rbf;
bias40_class_rbf = zeros(1,10);
bias40_class_lin = bias40_class_rbf;
Ai_Yi40_rbf = cell(1,10);
sup_vec_idx40_rbf = cell(1,10);

weight80_class_rbf = zeros(size(pc80,2),10);
weight80_class_lin = weight80_class_rbf;
bias80_class_rbf = zeros(1,10);
bias80_class_lin = bias80_class_rbf;
Ai_Yi80_rbf = cell(1,10);
sup_vec_idx80_rbf = cell(1,10);

weight200_class_rbf = zeros(size(pc200,2),10);
weight200_class_lin = weight200_class_rbf;
bias200_class_rbf = zeros(1,10);
bias200_class_lin = bias200_class_rbf;
Ai_Yi200_rbf = cell(1,10);
sup_vec_idx200_rbf = cell(1,10);

h = waitbar(0,'Please wait...');
for softC=1:length(soft_margin_c)
    disp('---------------------------------------------------------------------');
    disp(['-----------@softmargin = ',num2str(soft_margin_c(softC)),'-----------']);
    for class_i=0:9
        % label exclusion for PCA40 (convert to binary label,i=1 non_i=-1)
        lab_exc_idx40 = lab_train~=class_i;
        lab_train40_exc(lab_exc_idx40,class_i+1) = -1;
        lab_exc_idx40 = find(lab_train==class_i);
        lab_train40_exc(lab_exc_idx40,class_i+1) = 1;
        
        lab_exc_idx_test40 = lab_test~=class_i;
        lab_test40_exc(lab_exc_idx_test40,class_i+1) = -1;
        lab_exc_idx_test40 = find(lab_test==class_i);
        lab_test40_exc(lab_exc_idx_test40,class_i+1) = 1;
        %---------------------------------------------------------------
        % label exclusion for PCA80 (convert to binary label,i=1 non_i=-1)
        lab_exc_idx80 = lab_train~=class_i;
        lab_train80_exc(lab_exc_idx80,class_i+1) = -1;
        lab_exc_idx80 = find(lab_train==class_i);
        lab_train80_exc(lab_exc_idx80,class_i+1) = 1;
        
        lab_exc_idx_test80 = lab_test~=class_i;
        lab_test80_exc(lab_exc_idx_test80,class_i+1) = -1;
        lab_exc_idx_test80 = find(lab_test==class_i);
        lab_test80_exc(lab_exc_idx_test80,class_i+1) = 1;
        %---------------------------------------------------------------
        % label exclusion for PCA200 (convert to binary label,i=1 non_i=-1)
        lab_exc_idx200 = lab_train~=class_i;
        lab_train200_exc(lab_exc_idx200,class_i+1) = -1;
        lab_exc_idx200 = find(lab_train==class_i);
        lab_train200_exc(lab_exc_idx200,class_i+1) = 1;
        
        lab_exc_idx_test200 = lab_test~=class_i;
        lab_test200_exc(lab_exc_idx_test200,class_i+1) = -1;
        lab_exc_idx_test200 = find(lab_test==class_i);
        lab_test200_exc(lab_exc_idx_test200,class_i+1) = 1;
        
        % SVM algorithm using data with 40 dim PCA compression and linear kernel
        disp(['Calculating SVM40 for linear kernel @class ',num2str(class_i+1)]);
        SVM40_model_class_lin = fitcsvm(xpc40',lab_train40_exc(:,class_i+1),'Solver','L1QP','BoxConstraint',soft_margin_c(softC),'KernelFunction','linear');
        sup_vec_idx40_lin = SVM40_model_class_lin.IsSupportVector==1;
        Ai_Yi40_lin = SVM40_model_class_lin.Alpha.*SVM40_model_class_lin.SupportVectorLabels;
        weight40_class_lin(:,class_i+1) = xpc40(:,sup_vec_idx40_lin)*Ai_Yi40_lin;
        bias40_class_lin(class_i+1) = SVM40_model_class_lin.Bias;
        
        % SVM algorithm using data with 40 dim PCA compression and radial kernel
        disp(['Calculating SVM40 for radial kernel @class ',num2str(class_i+1)]);
        SVM40_model_class_rbf = fitcsvm(xpc40',lab_train40_exc(:,class_i+1),'Solver','L1QP','BoxConstraint',soft_margin_c(softC),'KernelFunction','RBF','KernelScale', rbf_sigma);
        sup_vec_idx40_rbf{class_i+1} = find(SVM40_model_class_rbf.IsSupportVector==1);
        Ai_Yi40_rbf{class_i+1} = SVM40_model_class_rbf.Alpha.*SVM40_model_class_rbf.SupportVectorLabels;
        bias40_class_rbf(class_i+1) = SVM40_model_class_rbf.Bias;
        
        % SVM algorithm using data with 80 dim PCA compression and linear kernel
        disp(['Calculating SVM80 for linear kernel @class ',num2str(class_i+1)]);
        SVM80_model_class_lin = fitcsvm(xpc80',lab_train80_exc(:,class_i+1),'Solver','L1QP','BoxConstraint',soft_margin_c(softC),'KernelFunction','linear');
        sup_vec_idx80_lin = SVM80_model_class_lin.IsSupportVector==1;
        Ai_Yi80_lin = SVM80_model_class_lin.Alpha.*SVM80_model_class_lin.SupportVectorLabels;
        weight80_class_lin(:,class_i+1) = xpc80(:,sup_vec_idx80_lin)*Ai_Yi80_lin;
        bias80_class_lin(class_i+1) = SVM80_model_class_lin.Bias;
        
        % SVM algorithm using data with 80 dim PCA compression and radial kernel
        disp(['Calculating SVM80 for radial kernel @class ',num2str(class_i+1)]);
        SVM80_model_class_rbf = fitcsvm(xpc80',lab_train80_exc(:,class_i+1),'Solver','L1QP','BoxConstraint',soft_margin_c(softC),'KernelFunction','RBF','KernelScale', rbf_sigma);
        sup_vec_idx80_rbf{class_i+1} = find(SVM80_model_class_rbf.IsSupportVector==1);
        Ai_Yi80_rbf{class_i+1} = SVM80_model_class_rbf.Alpha.*SVM80_model_class_rbf.SupportVectorLabels;
        bias80_class_rbf(class_i+1) = SVM80_model_class_rbf.Bias;
        
        % SVM algorithm using data with 200 dim PCA compression and linear kernel
        disp(['Calculating SVM200 for linear kernel @class ',num2str(class_i+1)]);
        SVM200_model_class_lin = fitcsvm(xpc200',lab_train200_exc(:,class_i+1),'Solver','L1QP','BoxConstraint',soft_margin_c(softC),'KernelFunction','linear');
        sup_vec_idx200_lin = SVM200_model_class_lin.IsSupportVector==1;
        Ai_Yi200_lin = SVM200_model_class_lin.Alpha.*SVM200_model_class_lin.SupportVectorLabels;
        weight200_class_lin(:,class_i+1) = xpc200(:,sup_vec_idx200_lin)*Ai_Yi200_lin;
        bias200_class_lin(class_i+1) = SVM200_model_class_lin.Bias;
        
        % SVM algorithm using data with 200 dim PCA compression and radial kernel
        disp(['Calculating SVM200 for radial kernel @class ',num2str(class_i+1)]);
        SVM200_model_class_rbf = fitcsvm(xpc200',lab_train200_exc(:,class_i+1),'Solver','L1QP','BoxConstraint',soft_margin_c(softC),'KernelFunction','RBF','KernelScale', rbf_sigma);
        sup_vec_idx200_rbf{class_i+1} = find(SVM200_model_class_rbf.IsSupportVector==1);
        Ai_Yi200_rbf{class_i+1} = SVM200_model_class_rbf.Alpha.*SVM200_model_class_rbf.SupportVectorLabels;
        bias200_class_rbf(class_i+1) = SVM200_model_class_rbf.Bias;
        
        waitbar((class_i+1)/10,h,sprintf('Calculate multiclass weight with multi softmargin...%2.1f%%',100*(class_i+1)/10));
    end
    
    %% evaluating deterministic function value
    %test data classification for PCA40
    test40_data_mat_lin = weight40_class_lin'*ypc40 + bias40_class_lin';
    dist_rbf40_mat = zeros(10,size(ypc40,2));
    for class_i=0:9
        for rbf_40=1:size(ypc40,2)
            dummy = xpc40(:,sup_vec_idx40_rbf{class_i+1}) - ypc40(:,rbf_40);
            dummy = sum(dummy.*dummy);
            dummy = dummy/(rbf_sigma^2);
            dummy = exp((-1)*dummy);
            dummy = Ai_Yi40_rbf{class_i+1}.*dummy';
            dummy = sum(dummy) + bias40_class_rbf(class_i+1);
            dist_rbf40_mat(class_i+1,rbf_40) = dummy;
        end
    end
    
    %test data classification for PCA80
    test80_data_mat_lin = weight80_class_lin'*ypc80 + bias80_class_lin';
    dist_rbf80_mat = zeros(10,size(ypc80,2));
    for class_i=0:9
        for rbf_80=1:size(ypc80,2)
            dummy = xpc80(:,sup_vec_idx80_rbf{class_i+1}) - ypc80(:,rbf_80);
            dummy = sum(dummy.*dummy);
            dummy = dummy/(rbf_sigma^2);
            dummy = exp((-1)*dummy);
            dummy = Ai_Yi80_rbf{class_i+1}.*dummy';
            dummy = sum(dummy) + bias80_class_rbf(class_i+1);
            dist_rbf80_mat(class_i+1,rbf_80) = dummy;
        end
    end
    
    %test data classification for PCA200
    test200_data_mat_lin = weight200_class_lin'*ypc200 + bias200_class_lin';
    dist_rbf200_mat = zeros(10,size(ypc200,2));
    for class_i=0:9
        for rbf_200=1:size(ypc200,2)
            dummy = xpc200(:,sup_vec_idx200_rbf{class_i+1}) - ypc200(:,rbf_200);
            dummy = sum(dummy.*dummy);
            dummy = dummy/(rbf_sigma^2);
            dummy = exp((-1)*dummy);
            dummy = Ai_Yi200_rbf{class_i+1}.*dummy';
            dummy = sum(dummy) + bias200_class_rbf(class_i+1);
            dist_rbf200_mat(class_i+1,rbf_200) = dummy;
        end
    end
    
    
    %% PCA40 evaluation code
    %test label result, assigning class that put data to most positive region
    % for linear
    test40_label_lin = find(test40_data_mat_lin==max(test40_data_mat_lin));
    test40_label_lin = mod(test40_label_lin,10)-1;
    test40_label_lin(test40_label_lin==-1) = 9;
    
    % for radial
    test40_label_rbf = find(dist_rbf40_mat==max(dist_rbf40_mat));
    test40_label_rbf = mod(test40_label_rbf,10)-1;
    test40_label_rbf(test40_label_rbf==-1) = 9;
    
    %evaluating the classification label
    acc40_lin(softC) = sum(test40_label_lin==lab_test)*100/length(lab_test);
    acc40_rbf(softC) = sum(test40_label_rbf==lab_test)*100/length(lab_test);
    
    %% PCA80 evaluation code
    %test label result, assigning class that put data to most positive region
    % for linear
    test80_label_lin = find(test80_data_mat_lin==max(test80_data_mat_lin));
    test80_label_lin = mod(test80_label_lin,10)-1;
    test80_label_lin(test80_label_lin==-1) = 9;
    
    % for radial
    test80_label_rbf = find(dist_rbf80_mat==max(dist_rbf80_mat));
    test80_label_rbf = mod(test80_label_rbf,10)-1;
    test80_label_rbf(test80_label_rbf==-1) = 9;
    
    %evaluating the classification label
    acc80_lin(softC) = sum(test80_label_lin==lab_test)*100/length(lab_test);
    acc80_rbf(softC) = sum(test80_label_rbf==lab_test)*100/length(lab_test);
    
    
    %% PCA200 evaluation code
    %test label result, assigning class that put data to most positive region
    % for linear
    test200_label_lin = find(test200_data_mat_lin==max(test200_data_mat_lin));
    test200_label_lin = mod(test200_label_lin,10)-1;
    test200_label_lin(test200_label_lin==-1) = 9;
    
    % for radial
    test200_label_rbf = find(dist_rbf200_mat==max(dist_rbf200_mat));
    test200_label_rbf = mod(test200_label_rbf,10)-1;
    test200_label_rbf(test200_label_rbf==-1) = 9;
    
    %evaluating the classification label
    acc200_lin(softC) = sum(test200_label_lin==lab_test)*100/length(lab_test);
    acc200_rbf(softC) = sum(test200_label_rbf==lab_test)*100/length(lab_test);
    
end
%% ERROR DOCUMENTATION
disp('-------------------------------------------------------------------------------------');
disp('------Overall performance, row = PCAdim/kernel, col = softmargin------');
disp(['Error using PCA40 and linear kernel = ',num2str(acc40_lin),'%']);
disp(['Error using PCA40 and radial kernel = ',num2str(acc40_rbf),'%']);
disp(['Error using PCA80 and linear kernel = ',num2str(acc80_lin),'%']);
disp(['Error using PCA80 and radial kernel = ',num2str(acc80_rbf),'%']);
disp(['Error using PCA200 and linear kernel = ',num2str(acc200_lin),'%']);
disp(['Error using PCA200 and radial kernel = ',num2str(acc200_rbf),'%']);
close(h);



