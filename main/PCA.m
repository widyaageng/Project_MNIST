clear;clc;

%%
%------DATA STRUCTURE--------%

% training image data structure
im_train = (fread(fopen('train-images.idx3-ubyte','r'),inf,'uint8'));
im_train_header = cast(im_train(1:16),'uint8');
im_train_header = reshape(dec2hex(im_train_header).',[8 4])';
im_train_magic = hex2dec(im_train_header(1,:));
im_train_count = hex2dec(im_train_header(2,:));
im_train_row = hex2dec(im_train_header(3,:));
im_train_col = hex2dec(im_train_header(4,:));
im_train = reshape(im_train(17:size(im_train)),[28*28 im_train_count]);
clear im_train_header;

% training label data structure
lab_train = (fread(fopen('train-labels.idx1-ubyte','r'),inf,'uint8'));
lab_train_header = cast(lab_train(1:8),'uint8');
lab_train_header = reshape(dec2hex(lab_train_header).',[8 2])';
lab_train_magic = hex2dec(lab_train_header(1,:));
lab_train_count = hex2dec(lab_train_header(2,:));
lab_train = lab_train(9:size(lab_train));
clear lab_train_header;

% test image data structure
im_test = (fread(fopen('t10k-images.idx3-ubyte','r'),inf,'uint8'));
im_test_header = cast(im_test(1:16),'uint8');
im_test_header = reshape(dec2hex(im_test_header).',[8 4])';
im_test_magic = hex2dec(im_test_header(1,:));
im_test_count = hex2dec(im_test_header(2,:));
im_test_row = hex2dec(im_test_header(3,:));
im_test_col = hex2dec(im_test_header(4,:));
im_test = reshape(im_test(17:size(im_test)), [28*28 im_test_count]);
clear im_test_header

% test label data structure
lab_test = (fread(fopen('t10k-labels.idx1-ubyte','r'),inf,'uint8'));
lab_test_header = cast(lab_test(1:8),'uint8');
lab_test_header = reshape(dec2hex(lab_test_header).',[8 2])';
lab_test_magic = hex2dec(lab_test_header(1,:));
lab_test_count = hex2dec(lab_test_header(2,:));
lab_test = lab_test(9:size(lab_test));
clear lab_test_header;


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

%% 2D feature extraction
%-------------- 2 PCs Projection --------------
pc2= train_U(:,1:2);

%-------------- Projecting Data ---------------
xpc2 = {};
for i = 1:size(pc2,2)
    xpc2{i} = pc2(:,i)'*im_train;
end

%------Grouping training data based on its label------

% Sorting the label and its respective projected data column
temp_lab_train = lab_train;
[temp_lab_train,lab_idx] = sort(temp_lab_train);
train_dump = xpc2{1}(:,lab_idx(1));
train_dump2 = xpc2{2}(:,lab_idx(1));
for i = 2:length(lab_idx)
    train_dump = [train_dump xpc2{1}(lab_idx(i))];
    train_dump2 = [train_dump2 xpc2{2}(lab_idx(i))];
end
xpc2 = {train_dump train_dump2};
clear train_dump;clear train_dump2;

% Initiate distinct color for scatter plot
RGB = [0.3686 0.3098    0.6353;
    0.2005    0.5593    0.7380;
    0.4558    0.7897    0.6458;
    0.7616    0.9058    0.6299;
    0.9277    0.9583    0.6442;
    0.9820    0.9206    0.6372;
    0.9942    0.7583    0.4312;
    0.9684    0.4799    0.2723;
    0.8525    0.2654    0.3082;
    0.6196    0.0039    0.2588];


% Finding the index of last column data on each class and scatter plot it
last_train_datum = [0];
figure;hold;
for i=1:10
    last_train_datum = [last_train_datum find(temp_lab_train==(i-1), 1, 'last' )];
    scatter(xpc2{1}(last_train_datum(i)+1:last_train_datum(i+1)),xpc2{2}(last_train_datum(i)+1:last_train_datum(i+1)),2,RGB(i,:));
end
colormap(RGB);
col_handle = colorbar('Location', 'EastOutside', 'YTick',[1/20 1/20+1/10 1/20+2/10 1/20+3/10 1/20+4/10 1/20+5/10 1/20+6/10 1/20+7/10 1/20+8/10 1/20+9/10]);
col_handle.TickLabels = {'0','1','2','3','4','5','6','7','8','9'};
grid;
xlabel('1st PC');
ylabel('2nd PC');
hold off;

% Visualizing the eigenvectors
figure;
imshow(double(reshape(255*pc2(:,1)/max(pc2(:,1)),[28 28])));
figure;
imshow(double(reshape(255*pc2(:,2)/max(pc2(:,2)),[28 28])))

%% 3D feature extraction

%-------------- 3 PCs Projection --------------
pc3= train_U(:,1:3);

%-------------- Projecting Data ---------------
xpc3 = {};
for i = 1:size(pc3,2)
    xpc3{i} = pc3(:,i)'*im_train;
end

%------Grouping training data based on its label------

% Sorting the label and its respective projected data column
train_dump = xpc3{1}(:,lab_idx(1));
train_dump2 = xpc3{2}(:,lab_idx(1));
train_dump3 = xpc3{3}(:,lab_idx(1));
for i = 2:length(lab_idx)
    train_dump = [train_dump xpc3{1}(lab_idx(i))];
    train_dump2 = [train_dump2 xpc3{2}(lab_idx(i))];
    train_dump3 = [train_dump3 xpc3{3}(lab_idx(i))];
end
xpc3 = {train_dump train_dump2 train_dump3};
clear train_dump;clear train_dump2;clear train_dump3;clear lab_idx;

% Finding the index of last column data on each class and scatter plot it
last_train_datum = [0];
figure;
hold on;
for i=1:10
    last_train_datum = [last_train_datum find(temp_lab_train==(i-1), 1, 'last' )];
    scatter3(xpc3{1}(last_train_datum(i)+1:last_train_datum(i+1)),xpc3{2}(last_train_datum(i)+1:last_train_datum(i+1)),xpc3{3}(last_train_datum(i)+1:last_train_datum(i+1)),2,RGB(i,:));
end
colormap(RGB);
col_handle = colorbar('Location', 'EastOutside', 'YTick',[1/20 1/20+1/10 1/20+2/10 1/20+3/10 1/20+4/10 1/20+5/10 1/20+6/10 1/20+7/10 1/20+8/10 1/20+9/10]);
col_handle.TickLabels = {'0','1','2','3','4','5','6','7','8','9'};
grid;
view(3);
xlabel('1st PC');
ylabel('2nd PC');
zlabel('3rd PC');
hold off;

% Visualizing the eigenvectors
figure;
imshow(double(reshape(255*pc3(:,1)/max(pc3(:,1)),[28 28])));
figure;
imshow(double(reshape(255*pc3(:,2)/max(pc3(:,2)),[28 28])))
figure;
imshow(double(reshape(255*pc3(:,3)/max(pc3(:,3)),[28 28])))

%% PCA dimension reduction to 40
KNN = 28;

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

% Classifying 40PCs projected test data to existing pool of projected train
% data
h = waitbar(0,'Please wait...');
index = 0;
knn_label_40 = 0*lab_test;
for i=1:size(ypc40,2)
    dist40 = zeros(1,length(xpc40));
    for j=1:size(xpc40,2)
        dist40(j) = sqrt((ypc40(:,i)-xpc40(:,j))'*(ypc40(:,i)-xpc40(:,j)));
        index = index + 1;
    end
    [dist40, dist_idx40] = sort(dist40);
    dist_idx40 = dist_idx40(1:KNN);
    knn_label_40(i) = mode(lab_train(dist_idx40));
    waitbar(index/(length(ypc40)*length(xpc40)),h,sprintf('Classifying test image PC40...%2.1f%%',100*index/(length(ypc40)*length(xpc40))));
end
close(h);

%% PCA dimension reduction to 80
KNN = 28;

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

% Classifying 80PCs projected test data to existing pool of projected train
% data
h = waitbar(0,'Please wait...');
index = 0;
knn_label_80 = 0*lab_test;
for i=1:size(ypc80,2)
    dist80 = zeros(1,length(xpc80));
    for j=1:size(xpc80,2)
        dist80(j) = sqrt((ypc80(:,i)-xpc80(:,j))'*(ypc80(:,i)-xpc80(:,j)));
        index = index + 1;
    end
    [dist80, dist_idx80] = sort(dist80);
    dist_idx80 = dist_idx80(1:KNN);
    knn_label_80(i) = mode(lab_train(dist_idx80));
    waitbar(index/(length(ypc80)*length(xpc80)),h,sprintf('Classifying test image PC80...%2.1f%%',100*index/(length(ypc80)*length(xpc80))));
end
close(h);

%% PCA dimension reduction to 200
KNN = 28;
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

% Classifying 200 PCs projected test data to existing pool of projected train
% data
h = waitbar(0,'Please wait...');
index = 0;
knn_label_200 = 0*lab_test;
for i=1:size(ypc200,2)
    dist200 = zeros(1,length(xpc200));
    for j=1:size(xpc200,2)
        dist200(j) = sqrt((ypc200(:,i)-xpc200(:,j))'*(ypc200(:,i)-xpc200(:,j)));
        index = index + 1;
    end
    [dist200, dist_idx200] = sort(dist200);
    dist_idx200 = dist_idx200(1:KNN);
    knn_label_200(i) = mode(lab_train(dist_idx200));
    waitbar(index/(length(ypc200)*length(xpc200)),h,sprintf('Classifying test image PC200...%2.1f%%',100*index/(length(ypc200)*length(xpc200))));
end
close(h);

%% PCA dimension reduction to number of PCs that have energy over 95%

threshold = (0.01:0.01:1);
track_d = 0*(1:1:100);
for i=1:length(threshold)
    eigen_energy = 0;
    while eigen_energy < threshold(i)
        track_d(i) = track_d(i) + 1;
        eigen_energy = sum(train_eig(1:track_d(i)))/sum(train_eig);
    end
end

figure;hold on;
plot(threshold,track_d,'.-r','LineWidth',1.5);
xlabel('total energy preserved');
ylabel('number of PCs');
grid;
title('Energy preserved vs #PCs');

d = track_d(threshold==0.95);

%% PCA dimension reduction to d95%
KNN = 28;
%-------------- 200 PCs Projection --------------
pcd = train_U(:,1:d);

%-------------- Projecting Data ---------------
xpcd = {}; % projected train image
ypcd = {}; % projected test image
for i = 1:size(pcd,2)
    xpcd{i} = pcd(:,i)'*im_train;
    ypcd{i} = pcd(:,i)'*im_test;
end
xpcd = cell2mat(xpcd');
ypcd = cell2mat(ypcd');

% Classifying 200 PCs projected test data to existing pool of projected train
% data
h = waitbar(0,'Please wait...');
index = 0;
knn_label_d = 0*lab_test;
for i=1:size(ypcd,2)
    distd = zeros(1,length(xpcd));
    for j=1:size(xpcd,2)
        distd(j) = sqrt((ypcd(:,i)-xpcd(:,j))'*(ypcd(:,i)-xpcd(:,j)));
        index = index + 1;
    end
    [distd, dist_idx] = sort(distd);
    dist_idx = dist_idx(1:KNN);
    knn_label_d(i) = mode(lab_train(dist_idx));
    waitbar(index/(length(ypcd)*length(xpcd)),h,sprintf('Classifying test image PCd...%2.1f%%',100*index/(length(ypcd)*length(xpcd))));
end
close(h);

%% ERROR DOCUMENTATION

class_acc_40 = (sum(lab_test==knn_label_40))*100/length(lab_test);
class_acc_80 = (sum(lab_test==knn_label_80))*100/length(lab_test);
class_acc_200 = (sum(lab_test==knn_label_200))*100/length(lab_test);
class_acc_d = (sum(lab_test==knn_label_d))*100/length(lab_test);
disp(['Error d@40 = ',num2str(class_acc_40),'%']);
disp(['Error d@80 = ',num2str(class_acc_80),'%']);
disp(['Error d@200 = ',num2str(class_acc_200),'%']);
disp(['Error d@95% energy = ',num2str(class_acc_d),'%', ' @d = ',num2str(d)]);



