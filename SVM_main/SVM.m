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
im_train = im_train(:,1:30);
lab_train = lab_train(1:30);
im_test = im_test(:,1:5);
lab_test = lab_test(1:5);
im_train_count = 30;
im_test_count = 5;



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
rad_sig_sq = [1000];

% capturing accuracy for linear kernel,for different soft margin
test_accuracy_lin = zeros(10,length(soft_margin_c));

% capturing accuracy for radial kernel, row=softmargin, col=radial spread
test_accuracy_rad = zeros(10,length(soft_margin_c),length(rad_sig_sq));

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

% SVM for projected data to 40 PCA dimension
% Finding soft hyperplane for the 10 class data
lab_train40_exc = lab_train*zeros(1,10);
lab_test40_exc = lab_test*zeros(1,10);

% Storing optimum weights W for each class(col) and softmargin(3rd dim), for
% linear kernel
soft_optimum_hyperplane_lin = zeros(im_train_count,10,length(soft_margin_c));

% soft offset
sot_b0_lin = zeros(10,length(soft_margin_c));

% Storing optimum weights W for each class(col), softmargin(3rd dim), spread (4th dim) for
% radial kernel
soft_optimum_hyperplane_rad = zeros(im_train_count, 10,length(soft_margin_c),length(rad_sig_sq));

for class_i=0:9
    % label exclusion (convert to binary label,i=1 non_i=-1)
    lab_exc_idx = lab_train~=class_i;
    lab_train40_exc(lab_exc_idx,class_i+1) = -1;
    lab_exc_idx = find(lab_train==class_i);
    lab_train40_exc(lab_exc_idx,class_i+1) = 1;
    
    lab_exc_idx_test = lab_test~=class_i;
    lab_test40_exc(lab_exc_idx_test,class_i+1) = -1;
    lab_exc_idx_test = find(lab_test==class_i);
    lab_test40_exc(lab_exc_idx_test,class_i+1) = 1;
    
    %optimoptions for quadprog
    optimoptions = optimset('LargeScale','off','MaxIter',10000,'Display','final');
    for c=1:length(soft_margin_c)
        disp(['Data PCA @',num2str(size(pc40,2)),';Linear Kernel with C=',num2str(soft_margin_c(c))]);
        disp(['----@class',num2str(class_i),'----']);
        %quadprog argument
        Aeq = transpose(lab_train40_exc(:,class_i+1));
        beq = 0;
        lb = zeros(size(lab_train40_exc(:,class_i+1),1),1);
        ub = soft_margin_c(c)*ones(size(lab_train40_exc(:,class_i+1),1),1);
        f = (-1)*(ones(size(lab_train40_exc(:,class_i+1),1),1));
        A = [];
        b = [];
        
        %% linear kernel
        midhess = xpc40'*xpc40(:,1);
        h = waitbar(0,'Please wait...');
        index = 0;
        for hess=2:size(xpc40,2)
            midhess = [midhess xpc40'*xpc40(:,hess)];
            waitbar(hess/(size(xpc40,2)),h,sprintf('Calculate linear Hessian...%2.1f%%',100*hess/(size(xpc40,2))));
        end
        H = midhess.*lab_train40_exc(:,class_i+1);
        H = H.*lab_train40_exc(:,class_i+1)';
        close(h);
        
        % Lagrangian optimisation using quadprog for linear kernel
        soft_optimum_hyperplane_lin(:,class_i+1,c) = quadprog(H,f,A,b,Aeq,beq,lb,ub,[],optimoptions);
        
        % alpha threshold to calculate b0
        th = soft_margin_c(c)*1e-8;
        
        % thresholding for alpha
        alpha_dump = soft_optimum_hyperplane_lin(:,class_i+1,c);
        alpha_pass_threshold_idx = find(alpha_dump>th);
        
        %averaging the constants b over the train data whose alpha values passed the threshold based on formula
        %b = 1/label - sum(alpha(i)*d(i)*K(x(i)x(sample)))
        temp_b = 0;
        b_increment = 0;
        dump_label = lab_train40_exc(:,class_i+1);
        for dataN=1:size(alpha_pass_threshold_idx,1)
            lin_kernel = xpc40(:,alpha_pass_threshold_idx(dataN))'*xpc40(:,alpha_pass_threshold_idx(dataN));
            b_increment = (1/lab_train40_exc(alpha_pass_threshold_idx(dataN))) -  dump_label(alpha_pass_threshold_idx(dataN))*(alpha_dump(alpha_pass_threshold_idx(dataN)))*lin_kernel;
            temp_b = temp_b + b_increment;
        end
        soft_b0_lin(class_i+1,c) = temp_b/size(alpha_pass_threshold_idx,1);
        
        %assigning label to testdata based on optimized hyperplane
        testlabel_with_weight = zeros(size(lab_test40_exc(:,class_i+1),1),1);
        for i=1:length(testlabel_with_weight)
            %calculating discriminant that has polynom kernel inside
            lin_kernel = xpc40'*ypc40(:,i);
            temp_gx = (lab_train40_exc(:,class_i+1))'*diag(soft_optimum_hyperplane_lin(:,class_i+1,c))*lin_kernel + soft_b0_lin(class_i+1,c);
            if temp_gx > 0
                testlabel_with_weight(i) = class_i;
            else
                testlabel_with_weight(i) = -1;
            end
        end
        
        % capturing the counts of correctly labelled test data for each
        % class and each softmargin
        test_accuracy_lin(class_i+1,c) = sum(and(testlabel_with_weight==1,testlabel_with_weight==lab_test40_exc(:,class_i+1)));
        
        
        %% Non-linear kernel
        % radial basis kernel
        
        % alpha threshold to calculate b0
        th_rad = soft_margin_c(c)*1e-8;
        dim = size(xpc40,1);
        
        disp(['Data PCA @',num2str(size(pc40,2)),';Radial Kernel with C=',num2str(soft_margin_c(c))]);
        disp(['----@class',num2str(class_i),'----']);
        
        h = waitbar(0,'Please wait...');
        index = 0;
        for kern=1:length(rad_sig_sq)
            
            midhess_rad = zeros(size(xpc40,2));
            H_rad = 0;
            for hess_c1=1:size(xpc40,2)
                for hess_c2=1:size(xpc40,2)
                    index = index + 1;
                    midhess_rad(hess_c1,hess_c2) = (-1)*(norm(xpc40(:,hess_c1) - xpc40(:,hess_c2))^2)/rad_sig_sq(kern);
                    waitbar(index/(length(rad_sig_sq)*size(xpc40,2)^2),h,sprintf('Calculate radial Hessian for each spread/var...%2.1f%%',100*index/(length(rad_sig_sq)*size(xpc40,2)^2)));
                end
            end
            midhess_rad = exp(midhess_rad);
            rad_H(:,:,kern) = midhess_rad.*lab_train40_exc(:,class_i+1);
            rad_H(:,:,kern) = rad_H(:,:,kern).*lab_train40_exc(:,class_i+1)';
            
            % Lagrangian optimisation using quadprog for radial kernel
            soft_optimum_hyperplane_rad(:,class_i+1,c,kern) = quadprog(rad_H(:,:,kern) ,f,A,b,Aeq,beq,lb,ub,[],optimoptions);
            
            % thresholding for alpha
            alpha_dump_rad = soft_optimum_hyperplane_rad(:,class_i+1,c,kern);
            alpha_pass_threshold_idx_rad = find(alpha_dump_rad>th_rad);
            
            %averaging the constants b over the train data whose alpha values passed the threshold based on formula
            %b = 1/label - sum(alpha(i)*d(i)*K(x(i)x(sample)))
            temp_b_rad = 0;
            b_increment_rad = 0;
            dump_label = lab_train40_exc(:,class_i+1);
            for dataN=1:size(alpha_pass_threshold_idx_rad,1)
                rad_kernel = xpc40(:,alpha_pass_threshold_idx_rad(dataN))'*xpc40(:,alpha_pass_threshold_idx_rad(dataN));
                b_increment_rad = (1/lab_train40_exc(alpha_pass_threshold_idx_rad(dataN))) -  dump_label(alpha_pass_threshold_idx_rad(dataN))*(alpha_dump_rad(alpha_pass_threshold_idx_rad(dataN)))*rad_kernel;
                temp_b_rad = temp_b_rad + b_increment_rad;
            end
            soft_b0_rad(class_i+1,c) = temp_b_rad/size(alpha_pass_threshold_idx_rad,1);
            
            %assigning label to testdata based on optimized hyperplane
            testlabel_with_weight_rad = zeros(size(lab_test40_exc(:,class_i+1),1),1);
            for i=1:length(testlabel_with_weight_rad)
                %calculating discriminant that has polynom kernel inside
                rad_kernel = xpc40'*ypc40(:,i);
                temp_gx_rad = (lab_train40_exc(:,class_i+1))'*diag(soft_optimum_hyperplane_rad(:,class_i+1,c,kern))*rad_kernel + soft_b0_rad(class_i+1,c);
                if temp_gx_rad > 0
                    testlabel_with_weight_rad(i) = 1;
                else
                    testlabel_with_weight_rad(i) = -1;
                end
            end
            
            % capturing the counts of correctly labelled test data for each
            % class and each softmargin
            test_accuracy_rad(class_i+1,c,kern) = sum(and(testlabel_with_weight_rad==1,testlabel_with_weight_rad==lab_test40_exc(:,class_i+1)));
        end
        close(h);
    end
end

%% DOCUMENTATION FOR PCA40

class_acc_40_lin = test_accuracy_lin;
class_acc_40_rad = test_accuracy_rad;




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

% SVM for projected data to 80 PCA dimension
% Finding soft hyperplane for the 10 class data
lab_train80_exc = lab_train*zeros(1,10);
lab_test80_exc = lab_test*zeros(1,10);

% Storing optimum weights W for each class(col) and softmargin(3rd dim), for
% linear kernel
soft_optimum_hyperplane_lin = zeros(im_train_count,10,length(soft_margin_c));

% soft offset
sot_b0_lin = zeros(10,length(soft_margin_c));

% Storing optimum weights W for each class(col), softmargin(3rd dim), spread (4th dim) for
% radial kernel
soft_optimum_hyperplane_rad = zeros(im_train_count, 10,length(soft_margin_c),length(rad_sig_sq));

for class_i=0:9
    % label exclusion (convert to binary label,i=1 non_i=-1)
    lab_exc_idx = lab_train~=class_i;
    lab_train80_exc(lab_exc_idx,class_i+1) = -1;
    lab_exc_idx = find(lab_train==class_i);
    lab_train80_exc(lab_exc_idx,class_i+1) = 1;
    
    lab_exc_idx_test = lab_test~=class_i;
    lab_test80_exc(lab_exc_idx_test,class_i+1) = -1;
    lab_exc_idx_test = find(lab_test==class_i);
    lab_test80_exc(lab_exc_idx_test,class_i+1) = 1;
    
    %optimoptions for quadprog
    optimoptions = optimset('LargeScale','off','MaxIter',10000,'Display','final');
    for c=1:length(soft_margin_c)
        %quadprog argument
        
        disp(['Data PCA @',num2str(size(pc80,2)),';Linear Kernel with C=',num2str(soft_margin_c(c))]);
        disp(['----@class',num2str(class_i),'----']);
        
        Aeq = transpose(lab_train80_exc(:,class_i+1));
        beq = 0;
        lb = zeros(size(lab_train80_exc(:,class_i+1),1),1);
        ub = soft_margin_c(c)*ones(size(lab_train80_exc(:,class_i+1),1),1);
        f = (-1)*(ones(size(lab_train80_exc(:,class_i+1),1),1));
        A = [];
        b = [];
        
        %% linear kernel
        midhess = xpc80'*xpc80(:,1);
        h = waitbar(0,'Please wait...');
        index = 0;
        for hess=2:size(xpc80,2)
            midhess = [midhess xpc80'*xpc80(:,hess)];
            waitbar(hess/(size(xpc80,2)),h,sprintf('Calculate linear Hessian...%2.1f%%',100*hess/(size(xpc80,2))));
        end
        H = midhess.*lab_train80_exc(:,class_i+1);
        H = H.*lab_train80_exc(:,class_i+1)';
        close(h);
        
        % Lagrangian optimisation using quadprog for linear kernel
        soft_optimum_hyperplane_lin(:,class_i+1,c) = quadprog(H,f,A,b,Aeq,beq,lb,ub,[],optimoptions);
        
        % alpha threshold to calculate b0
        th = soft_margin_c(c)*1e-8;
        
        % thresholding for alpha
        alpha_dump = soft_optimum_hyperplane_lin(:,class_i+1,c);
        alpha_pass_threshold_idx = find(alpha_dump>th);
        
        %averaging the constants b over the train data whose alpha values passed the threshold based on formula
        %b = 1/label - sum(alpha(i)*d(i)*K(x(i)x(sample)))
        temp_b = 0;
        b_increment = 0;
        dump_label = lab_train80_exc(:,class_i+1);
        for dataN=1:size(alpha_pass_threshold_idx,1)
            lin_kernel = xpc80(:,alpha_pass_threshold_idx(dataN))'*xpc80(:,alpha_pass_threshold_idx(dataN));
            b_increment = (1/lab_train80_exc(alpha_pass_threshold_idx(dataN))) -  dump_label(alpha_pass_threshold_idx(dataN))*(alpha_dump(alpha_pass_threshold_idx(dataN)))*lin_kernel;
            temp_b = temp_b + b_increment;
        end
        soft_b0_lin(class_i+1,c) = temp_b/size(alpha_pass_threshold_idx,1);
        
        %assigning label to testdata based on optimized hyperplane
        testlabel_with_weight = zeros(size(lab_test80_exc(:,class_i+1),1),1);
        for i=1:length(testlabel_with_weight)
            %calculating discriminant that has polynom kernel inside
            lin_kernel = xpc80'*ypc80(:,i);
            temp_gx = (lab_train80_exc(:,class_i+1))'*diag(soft_optimum_hyperplane_lin(:,class_i+1,c))*lin_kernel + soft_b0_lin(class_i+1,c);
            if temp_gx > 0
                testlabel_with_weight(i) = 1;
            else
                testlabel_with_weight(i) = -1;
            end
        end
        
        % capturing the counts of correctly labelled test data for each
        % class and each softmargin
        test_accuracy_lin(class_i+1,c) = sum(and(testlabel_with_weight==1,testlabel_with_weight==lab_test80_exc(:,class_i+1)));
        
        
        %% Non-linear kernel
        % radial basis kernel
        
        % alpha threshold to calculate b0
        th_rad = soft_margin_c(c)*1e-8;
        dim = size(xpc80,1);
        
        disp(['Data PCA @',num2str(size(pc80,2)),';Radial Kernel with C=',num2str(soft_margin_c(c))]);
        disp(['----@class',num2str(class_i),'----']);
        
        h = waitbar(0,'Please wait...');
        index = 0;
        for kern=1:length(rad_sig_sq)
            
            midhess_rad = zeros(size(xpc80,2));
            H_rad = 0;
            for hess_c1=1:size(xpc80,2)
                for hess_c2=1:size(xpc80,2)
                    index = index + 1;
                    midhess_rad(hess_c1,hess_c2) = (-1)*(norm(xpc80(:,hess_c1) - xpc80(:,hess_c2))^2)/rad_sig_sq(kern);
                    waitbar(index/(length(rad_sig_sq)*size(xpc80,2)^2),h,sprintf('Calculate radial Hessian for each spread/var...%2.1f%%',100*index/(length(rad_sig_sq)*size(xpc80,2)^2)));
                end
            end
            midhess_rad = exp(midhess_rad);
            rad_H(:,:,kern) = midhess_rad.*lab_train80_exc(:,class_i+1);
            rad_H(:,:,kern) = rad_H(:,:,kern).*lab_train80_exc(:,class_i+1)';
            
            % Lagrangian optimisation using quadprog for radial kernel
            soft_optimum_hyperplane_rad(:,class_i+1,c,kern) = quadprog(rad_H(:,:,kern) ,f,A,b,Aeq,beq,lb,ub,[],optimoptions);
            
            % thresholding for alpha
            alpha_dump_rad = soft_optimum_hyperplane_rad(:,class_i+1,c,kern);
            alpha_pass_threshold_idx_rad = find(alpha_dump_rad>th_rad);
            
            %averaging the constants b over the train data whose alpha values passed the threshold based on formula
            %b = 1/label - sum(alpha(i)*d(i)*K(x(i)x(sample)))
            temp_b_rad = 0;
            b_increment_rad = 0;
            dump_label = lab_train80_exc(:,class_i+1);
            for dataN=1:size(alpha_pass_threshold_idx_rad,1)
                rad_kernel = xpc80(:,alpha_pass_threshold_idx_rad(dataN))'*xpc80(:,alpha_pass_threshold_idx_rad(dataN));
                b_increment_rad = (1/lab_train80_exc(alpha_pass_threshold_idx_rad(dataN))) -  dump_label(alpha_pass_threshold_idx_rad(dataN))*(alpha_dump_rad(alpha_pass_threshold_idx_rad(dataN)))*rad_kernel;
                temp_b_rad = temp_b_rad + b_increment_rad;
            end
            soft_b0_rad(class_i+1,c) = temp_b_rad/size(alpha_pass_threshold_idx_rad,1);
            
            %assigning label to testdata based on optimized hyperplane
            testlabel_with_weight_rad = zeros(size(lab_test80_exc(:,class_i+1),1),1);
            for i=1:length(testlabel_with_weight_rad)
                %calculating discriminant that has polynom kernel inside
                rad_kernel = xpc80'*ypc80(:,i);
                temp_gx_rad = (lab_train80_exc(:,class_i+1))'*diag(soft_optimum_hyperplane_rad(:,class_i+1,c,kern))*rad_kernel + soft_b0_rad(class_i+1,c);
                if temp_gx_rad > 0
                    testlabel_with_weight_rad(i) = 1;
                else
                    testlabel_with_weight_rad(i) = -1;
                end
            end
            
            % capturing the counts of correctly labelled test data for each
            % class and each softmargin
            test_accuracy_rad(class_i+1,c,kern) = sum(and(testlabel_with_weight_rad==1,testlabel_with_weight_rad==lab_test80_exc(:,class_i+1)));
        end
        close(h);
    end
end

%% DOCUMENTATION FOR PCA80

class_acc_80_lin = test_accuracy_lin;
class_acc_80_rad = test_accuracy_rad;



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

% Storing optimum weights W for each class(col) and softmargin(3rd dim), for
% linear kernel
soft_optimum_hyperplane_lin = zeros(im_train_count,10,length(soft_margin_c));

% soft offset
sot_b0_lin = zeros(10,length(soft_margin_c));

% Storing optimum weights W for each class(col), softmargin(3rd dim), spread (4th dim) for
% radial kernel
soft_optimum_hyperplane_rad = zeros(im_train_count, 10,length(soft_margin_c),length(rad_sig_sq));

for class_i=0:9
    % label exclusion (convert to binary label,i=1 non_i=-1)
    lab_exc_idx = lab_train~=class_i;
    lab_train200_exc(lab_exc_idx,class_i+1) = -1;
    lab_exc_idx = find(lab_train==class_i);
    lab_train200_exc(lab_exc_idx,class_i+1) = 1;
    
    lab_exc_idx_test = lab_test~=class_i;
    lab_test200_exc(lab_exc_idx_test,class_i+1) = -1;
    lab_exc_idx_test = find(lab_test==class_i);
    lab_test200_exc(lab_exc_idx_test,class_i+1) = 1;
    
    %optimoptions for quadprog
    optimoptions = optimset('LargeScale','off','MaxIter',10000,'Display','final');
    for c=1:length(soft_margin_c)
        
        disp(['Data PCA @',num2str(size(pc200,2)),';Linear Kernel with C=',num2str(soft_margin_c(c))]);
        disp(['----@class',num2str(class_i),'----']);
        
        %quadprog argument
        Aeq = transpose(lab_train200_exc(:,class_i+1));
        beq = 0;
        lb = zeros(size(lab_train200_exc(:,class_i+1),1),1);
        ub = soft_margin_c(c)*ones(size(lab_train200_exc(:,class_i+1),1),1);
        f = (-1)*(ones(size(lab_train200_exc(:,class_i+1),1),1));
        A = [];
        b = [];
        
        %% linear kernel
        midhess = xpc200'*xpc200(:,1);
        h = waitbar(0,'Please wait...');
        index = 0;
        for hess=2:size(xpc200,2)
            midhess = [midhess xpc200'*xpc200(:,hess)];
            waitbar(hess/(size(xpc200,2)),h,sprintf('Calculate linear Hessian...%2.1f%%',100*hess/(size(xpc200,2))));
        end
        H = midhess.*lab_train200_exc(:,class_i+1);
        H = H.*lab_train200_exc(:,class_i+1)';
        close(h);
        
        % Lagrangian optimisation using quadprog for linear kernel
        soft_optimum_hyperplane_lin(:,class_i+1,c) = quadprog(H,f,A,b,Aeq,beq,lb,ub,[],optimoptions);
        
        % alpha threshold to calculate b0
        th = soft_margin_c(c)*1e-8;
        
        % thresholding for alpha
        alpha_dump = soft_optimum_hyperplane_lin(:,class_i+1,c);
        alpha_pass_threshold_idx = find(alpha_dump>th);
        
        %averaging the constants b over the train data whose alpha values passed the threshold based on formula
        %b = 1/label - sum(alpha(i)*d(i)*K(x(i)x(sample)))
        temp_b = 0;
        b_increment = 0;
        dump_label = lab_train200_exc(:,class_i+1);
        for dataN=1:size(alpha_pass_threshold_idx,1)
            lin_kernel = xpc200(:,alpha_pass_threshold_idx(dataN))'*xpc200(:,alpha_pass_threshold_idx(dataN));
            b_increment = (1/lab_train200_exc(alpha_pass_threshold_idx(dataN))) -  dump_label(alpha_pass_threshold_idx(dataN))*(alpha_dump(alpha_pass_threshold_idx(dataN)))*lin_kernel;
            temp_b = temp_b + b_increment;
        end
        soft_b0_lin(class_i+1,c) = temp_b/size(alpha_pass_threshold_idx,1);
        
        %assigning label to testdata based on optimized hyperplane
        testlabel_with_weight = zeros(size(lab_test200_exc(:,class_i+1),1),1);
        for i=1:length(testlabel_with_weight)
            %calculating discriminant that has polynom kernel inside
            lin_kernel = xpc200'*ypc200(:,i);
            temp_gx = (lab_train200_exc(:,class_i+1))'*diag(soft_optimum_hyperplane_lin(:,class_i+1,c))*lin_kernel + soft_b0_lin(class_i+1,c);
            if temp_gx > 0
                testlabel_with_weight(i) = 1;
            else
                testlabel_with_weight(i) = -1;
            end
        end
        
        % capturing the counts of correctly labelled test data for each
        % class and each softmargin
        test_accuracy_lin(class_i+1,c) = sum(and(testlabel_with_weight==1,testlabel_with_weight==lab_test200_exc(:,class_i+1)));
        
        
        %% Non-linear kernel
        % radial basis kernel
        
        % alpha threshold to calculate b0
        th_rad = soft_margin_c(c)*1e-8;
        dim = size(xpc200,1);
        
        disp(['Data PCA @',num2str(size(pc200,2)),';Radial Kernel with C=',num2str(soft_margin_c(c))]);
        disp(['----@class',num2str(class_i),'----']);
        
        h = waitbar(0,'Please wait...');
        index = 0;
        for kern=1:length(rad_sig_sq)
            
            midhess_rad = zeros(size(xpc200,2));
            H_rad = 0;
            for hess_c1=1:size(xpc200,2)
                for hess_c2=1:size(xpc200,2)
                    index = index + 1;
                    midhess_rad(hess_c1,hess_c2) = (-1)*(norm(xpc200(:,hess_c1) - xpc200(:,hess_c2))^2)/rad_sig_sq(kern);
                    waitbar(index/(length(rad_sig_sq)*size(xpc200,2)^2),h,sprintf('Calculate radial Hessian for each spread/var...%2.1f%%',100*index/(length(rad_sig_sq)*size(xpc200,2)^2)));
                end
            end
            midhess_rad = exp(midhess_rad);
            rad_H(:,:,kern) = midhess_rad.*lab_train200_exc(:,class_i+1);
            rad_H(:,:,kern) = rad_H(:,:,kern).*lab_train200_exc(:,class_i+1)';
            
            % Lagrangian optimisation using quadprog for radial kernel
            soft_optimum_hyperplane_rad(:,class_i+1,c,kern) = quadprog(rad_H(:,:,kern) ,f,A,b,Aeq,beq,lb,ub,[],optimoptions);
            
            % thresholding for alpha
            alpha_dump_rad = soft_optimum_hyperplane_rad(:,class_i+1,c,kern);
            alpha_pass_threshold_idx_rad = find(alpha_dump_rad>th_rad);
            
            %averaging the constants b over the train data whose alpha values passed the threshold based on formula
            %b = 1/label - sum(alpha(i)*d(i)*K(x(i)x(sample)))
            temp_b_rad = 0;
            b_increment_rad = 0;
            dump_label = lab_train200_exc(:,class_i+1);
            for dataN=1:size(alpha_pass_threshold_idx_rad,1)
                rad_kernel = xpc200(:,alpha_pass_threshold_idx_rad(dataN))'*xpc200(:,alpha_pass_threshold_idx_rad(dataN));
                b_increment_rad = (1/lab_train200_exc(alpha_pass_threshold_idx_rad(dataN))) -  dump_label(alpha_pass_threshold_idx_rad(dataN))*(alpha_dump_rad(alpha_pass_threshold_idx_rad(dataN)))*rad_kernel;
                temp_b_rad = temp_b_rad + b_increment_rad;
            end
            soft_b0_rad(class_i+1,c) = temp_b_rad/size(alpha_pass_threshold_idx_rad,1);
            
            %assigning label to testdata based on optimized hyperplane
            testlabel_with_weight_rad = zeros(size(lab_test200_exc(:,class_i+1),1),1);
            for i=1:length(testlabel_with_weight_rad)
                %calculating discriminant that has polynom kernel inside
                rad_kernel = xpc200'*ypc200(:,i);
                temp_gx_rad = (lab_train200_exc(:,class_i+1))'*diag(soft_optimum_hyperplane_rad(:,class_i+1,c,kern))*rad_kernel + soft_b0_rad(class_i+1,c);
                if temp_gx_rad > 0
                    testlabel_with_weight_rad(i) = 1;
                else
                    testlabel_with_weight_rad(i) = -1;
                end
            end
            
            % capturing the counts of correctly labelled test data for each
            % class and each softmargin
            test_accuracy_rad(class_i+1,c,kern) = sum(and(testlabel_with_weight_rad==1,testlabel_with_weight_rad==lab_test200_exc(:,class_i+1)));
        end
        close(h);
    end
end

%% DOCUMENTATION FOR PCA80

class_acc_200_lin = test_accuracy_lin;
class_acc_200_rad = test_accuracy_rad;


%% ERROR DOCUMENTATION

% class_acc_40 = (sum(lab_test==knn_label_40))*100/length(lab_test);
% class_acc_80 = (sum(lab_test==knn_label_80))*100/length(lab_test);
% class_acc_200 = (sum(lab_test==knn_label_200))*100/length(lab_test);
% class_acc_d = (sum(lab_test==knn_label_d))*100/length(lab_test);
% disp(['Error d@40 = ',num2str(class_acc_40),'%']);
% disp(['Error d@80 = ',num2str(class_acc_80),'%']);
% disp(['Error d@200 = ',num2str(class_acc_200),'%']);
% disp(['Error d@95% energy = ',num2str(class_acc_d),'%', ' @d = ',num2str(d)]);

